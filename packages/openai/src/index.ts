import { deepMerge } from '@mi-gpt/utils';
import type { Prettify } from '@mi-gpt/utils/typing';
import OpenAIClient from 'openai';
import type { RequestOptions } from 'openai/core';
import type { ChatCompletionCreateParamsBase } from 'openai/resources/chat/completions';
import { ProxyAgent } from 'proxy-agent';
import { type OpenAIConfig, kDefaultOpenAIConfig } from './config.js';
import { parseWebSearchResponse, shouldUseWebSearch } from './web-search.js';

export {
  parseWebSearchResponse,
  sanitizeWebSearchText,
  shouldUseWebSearch,
} from './web-search.js';
export type { WebSearchResult, WebSearchSource } from './web-search.js';

class _OpenAI {
  private _client?: OpenAIClient;
  private _abortCallbacks: Record<string, VoidFunction> = {};

  config: OpenAIConfig = {};

  init(config?: OpenAIConfig) {
    this.config = deepMerge(kDefaultOpenAIConfig, config);
    this._client ??= new OpenAIClient({
      baseURL: this.config.baseURL,
      apiKey: this.config.apiKey,
      httpAgent: this.config.enableProxy ? new ProxyAgent() : undefined,
      ...(this.config.extra?.clientOptions as any),
    });
  }

  dispose() {
    this._client = null as any;
    this._abortCallbacks = {};
  }

  cancel(requestId?: string) {
    if (requestId && this._abortCallbacks[requestId]) {
      this._abortCallbacks[requestId]();
      delete this._abortCallbacks[requestId];
    }
  }

  async chat(options: {
    requestId?: string;
    onStream?: (text: string) => void;
    onError?: (error: Error) => Promise<void>;
    requestOptions?: Prettify<RequestOptions>;
    createParams: Prettify<Partial<ChatCompletionCreateParamsBase>>;
  }) {
    const { requestId, onStream, requestOptions, createParams, onError } = options;

    let signal: AbortSignal | undefined;
    if (requestId) {
      const controller = new AbortController();
      this._abortCallbacks[requestId] = () => controller.abort();
      signal = controller.signal;
    }

    const params = deepMerge(
      {
        model: this.config.model,
        ...(this.config.extra?.createParams as any),
      },
      createParams,
    );

    const mergedRequestOptions = deepMerge(
      {
        ...(this.config.extra?.requestOptions as any),
      },
      { ...requestOptions, signal },
    );

    let fallbackNotice = '';
    const webSearchEnabled =
      !params.stream &&
      this.config.webSearch?.enabled &&
      (this.config.webSearch.strategy === 'auto' ||
        shouldUseWebSearch(params.messages as any[]));
    if (webSearchEnabled) {
      try {
        const response = await this._client!.responses.create(
          {
            model: params.model!,
            input: params.messages as any,
            // 仅使用默认的 search_engine（「联网资源」，每月免费 2 万次），
            // 不传 sources 即不会调用头条/抖音/墨迹天气等收费数据源（各 6 元/千次）。
            // max_keyword 限制单轮搜索的关键词数量，避免额度被多关键词放大消耗。
            tools: [{ type: 'web_search', max_keyword: 2 }],
            tool_choice: 'auto',
            // 限制一次回答内最多执行 2 轮工具调用（默认 3），进一步压低额度消耗上限。
            max_tool_calls: 2,
          } as any,
          mergedRequestOptions,
        );
        const result = parseWebSearchResponse(response);
        if (!result.text) {
          throw new Error('Responses API 未返回可播放的文字');
        }
        if (result.searched) {
          if (result.sources.length > 0) {
            console.log(
              '🌐 联网来源（引用链接，共 ',
              result.sources.length,
              ' 条）：',
              result.sources.map((source) => `${source.title}: ${source.url}`).join('\n'),
            );
          } else {
            console.log('🌐 已使用联网搜索');
          }
          // 打印真实计费用量：tool_usage_details 会按来源给出调用次数，
          // 其中 search_engine 即「联网资源」消耗次数，用于核对免费额度。
          const usage: any = (response as any)?.usage;
          if (usage?.tool_usage_details || usage?.tool_usage) {
            console.log(
              '📊 联网用量（计费口径）：',
              JSON.stringify(usage.tool_usage_details ?? usage.tool_usage),
            );
          }
        }
        if (requestId) {
          delete this._abortCallbacks[requestId];
        }
        return result.text;
      } catch (error) {
        if (signal?.aborted) {
          if (requestId) {
            delete this._abortCallbacks[requestId];
          }
          return '';
        }
        const message = error instanceof Error ? error.message : String(error);
        console.error(`❌ 联网搜索异常，将使用普通模型回答：${message}`);
        fallbackNotice = this.config.webSearch?.fallbackNotice ?? '';
      }
    }

    const res = await this._client!.chat.completions.create(
      params,
      mergedRequestOptions,
    ).catch(async (e) => {
      const message = e instanceof Error ? e.message : String(e);
      console.error(`❌ LLM 响应异常：${message}`);
      await onError?.(e);
      return null;
    });

    let result = '';

    if (params.stream) {
      for await (const chunk of (res ?? []) as any) {
        const text = chunk.choices[0]?.delta?.content || '';
        const aborted = requestId && !Object.keys(this._abortCallbacks).includes(requestId);
        if (aborted) {
          result = '';
          break;
        }
        if (text) {
          result += text;
          onStream?.(text);
        }
      }
    } else {
      result = res?.choices?.[0]?.message?.content ?? '';
      if (fallbackNotice && result) {
        result = `${fallbackNotice}${result}`;
      }
    }

    if (requestId) {
      delete this._abortCallbacks[requestId];
    }

    return result;
  }
}

export const OpenAI = new _OpenAI();
