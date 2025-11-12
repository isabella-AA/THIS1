from typing import Optional, Union, Generator, Dict, List
import os
#import openai
#import anthropic
#import requests
#from openai import OpenAI
from zhipuai import ZhipuAI  
import json


class LLMAPIHandler:
    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 ark_api_key: Optional[str] = None,
                 siliconflow_api_key: Optional[str] = None,
                 zhipuai_api_key: Optional[str] = None,
                 qwen_api_key: Optional[str] = None,
                 deepseek_api_key: Optional[str] = None):
        """
        Initialize API clients for different LLM providers
        """
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        if anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        if ark_api_key:
            self.ark_client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=ark_api_key,
            )
        if siliconflow_api_key:
            self.siliconflow_api_key = siliconflow_api_key
            self.siliconflow_base_url = "https://api.siliconflow.cn/v1"
        if zhipuai_api_key:
            self.zhipuai_client = ZhipuAI(api_key=zhipuai_api_key)
        if qwen_api_key:
            self.qwen_client = OpenAI(
                api_key=qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        if deepseek_api_key:
            self.deepseek_client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com"
            )

        # Default model configurations
        self.default_models = {
            'openai': 'gpt-4o-mini',
            'claude': 'claude-sonnet-4-20250514',
            'ark': 'doubao-1-5-lite-32k-250115',
            'siliconflow': 'deepseek-ai/DeepSeek-V3',
            'zhipuai': 'glm-4-air',
            'qwen': 'qwen-turbo-2025-04-28',
            'deepseek': 'deepseek-chat'
        }

    def call_deepseek(self,
                      prompt: str,
                      model: Optional[str] = None,
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      stream: bool = False,
                      system_prompt: Optional[str] = None,
                      history: Optional[List[Dict[str, str]]] = None) -> Union[str, Generator[str, None, None]]:
        """
        Call DeepSeek API with streaming support
        """
        try:
            model = model or self.default_models['deepseek']
            max_tokens = max_tokens or 1000
            temperature = temperature or 0.7

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 添加历史消息
            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": prompt})

            response = self.deepseek_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                def generate():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                return generate()
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API call failed: {e}")
            return f"API call failed: {str(e)}"

    def call_zhipuai(self,
                     prompt: str,
                     model: Optional[str] = None,
                     max_tokens: Optional[int] = None,
                     temperature: Optional[float] = None,
                     stream: bool = False,
                     system_prompt: Optional[str] = None,
                     history: Optional[List[Dict[str, str]]] = None) -> Union[str, Generator[str, None, None]]:
        """
        Call ZhipuAI API with streaming support
        """
        try:
            model = model or self.default_models['zhipuai']
            max_tokens = max_tokens or 1000
            temperature = temperature or 0.7

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 添加历史消息
            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": prompt})

            response = self.zhipuai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                def generate():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                return generate()
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(f"ZhipuAI API call failed: {e}")
            return f"API call failed: {str(e)}"

    def call_qwen(self,
                  prompt: str,
                  model: Optional[str] = None,
                  max_tokens: Optional[int] = None,
                  temperature: Optional[float] = None,
                  stream: bool = False,
                  system_prompt: Optional[str] = None,
                  history: Optional[List[Dict[str, str]]] = None) -> Union[str, Generator[str, None, None]]:
        """
        Call Qwen API with streaming support
        """
        try:
            model = model or self.default_models['qwen']
            max_tokens = max_tokens or 1000
            temperature = temperature or 0.7

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 添加历史消息
            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": prompt})

            response = self.qwen_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                def generate():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                return generate()
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(f"Qwen API call failed: {e}")
            return f"API call failed: {str(e)}"

    def call_siliconflow(self,
                         prompt: str,
                         model: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None,
                         stream: bool = False,
                         system_prompt: Optional[str] = None,
                         history: Optional[List[Dict[str, str]]] = None) -> Union[str, Generator[str, None, None]]:
        """
        Call SiliconFlow API with streaming support
        """
        try:
            model = model or self.default_models['siliconflow']
            max_tokens = max_tokens or 1000
            temperature = temperature or 0.7

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 添加历史消息
            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.7,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "stream": stream,
                "response_format": {"type": "text"}
            }

            headers = {
                "Authorization": f"Bearer {self.siliconflow_api_key}",
                "Content-Type": "application/json"
            }

            if stream:
                def generate():
                    response = requests.post(
                        f"{self.siliconflow_base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                        stream=True
                    )
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('data:'):
                                try:
                                    content = json.loads(decoded_line[5:].strip())
                                    if content and 'choices' in content and content['choices'] and 'delta' in \
                                            content['choices'][0] and 'content' in content['choices'][0]['delta']:
                                        yield content['choices'][0]['delta']['content']
                                except json.JSONDecodeError:
                                    continue

                return generate()
            else:
                response = requests.post(
                    f"{self.siliconflow_base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"SiliconFlow API call failed: {e}")
            return f"API call failed: {str(e)}"

    def call_claude(self,
                    prompt: str,
                    model: Optional[str] = None,
                    max_tokens: Optional[int] = None,
                    stream: bool = False,
                    system_prompt: Optional[str] = None,
                    history: Optional[List[Dict[str, str]]] = None) -> Union[str, Generator[str, None, None]]:
        """
        Call Claude API with streaming support
        """
        try:
            model = model or self.default_models['claude']
            max_tokens = max_tokens or 1000

            messages = []
            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": prompt})

            kwargs = {
                'model': model,
                'max_tokens': max_tokens,
                'messages': messages,
                'stream': stream
            }

            if system_prompt:
                kwargs['system'] = system_prompt

            if stream:
                def generate():
                    with self.anthropic_client.messages.stream(**kwargs) as stream:
                        for text in stream.text_stream:
                            yield text

                return generate()
            else:
                message = self.anthropic_client.messages.create(**kwargs)
                return message.content[0].text
        except Exception as e:
            print(f"Claude API call failed: {e}")
            return f"API call failed: {str(e)}"

    def call_openai(self,
                    prompt: str,
                    model: Optional[str] = None,
                    max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None,
                    stream: bool = False,
                    system_prompt: Optional[str] = None,
                    history: Optional[List[Dict[str, str]]] = None) -> Union[str, Generator[str, None, None]]:
        """
        Call OpenAI API with streaming support
        """
        try:
            model = model or self.default_models['openai']
            max_tokens = max_tokens or 1000
            temperature = temperature or 0.7

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 添加历史消息
            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": prompt})

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                def generate():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                return generate()
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return f"API call failed: {str(e)}"

    def call_ark(self,
                 prompt: str,
                 model: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 stream: bool = False,
                 system_prompt: Optional[str] = None,
                 history: Optional[List[Dict[str, str]]] = None) -> Union[str, Generator[str, None, None]]:
        """
        Call ARK API with streaming support
        """
        try:
            model = model or self.default_models['ark']
            max_tokens = max_tokens or 1000
            temperature = temperature or 0.7

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 添加历史消息
            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": prompt})

            response = self.ark_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                def generate():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                return generate()
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(f"ARK API call failed: {e}")
            return f"API call failed: {str(e)}"

    def call_llm(self,
                 provider: str,
                 prompt: str,
                 model: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 stream: bool = False,
                 system_prompt: Optional[str] = None,
                 history: Optional[List[Dict[str, str]]] = None) -> Union[str, Generator[str, None, None]]:
        """
        Unified method to call different LLM providers with streaming support
        """
        provider = provider.lower()

        # Set default parameters based on provider
        params = {
            'prompt': prompt,
            'model': model or self.default_models.get(provider),
            'max_tokens': max_tokens or 1000,
            'stream': stream,
            'system_prompt': system_prompt,
            'history': history
        }

        # Add temperature parameter if applicable
        if provider != 'claude':
            params['temperature'] = temperature or 0.7

        # Call provider-specific method
        if provider == 'openai':
            return self.call_openai(**params)
        elif provider == 'claude':
            return self.call_claude(**params)
        elif provider == 'ark':
            return self.call_ark(**params)
        elif provider == 'siliconflow':
            return self.call_siliconflow(**params)
        elif provider == 'zhipuai':
            return self.call_zhipuai(**params)
        elif provider == 'qwen':
            return self.call_qwen(**params)
        elif provider == 'deepseek':
            return self.call_deepseek(**params)
        else:
            return f"Unsupported provider: {provider}"


handler = LLMAPIHandler(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    ark_api_key=os.getenv("ARK_API_KEY"),
    siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY"),
    zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    qwen_api_key=os.getenv("QWEN_API_KEY"),
    deepseek_api_key=os.getenv("DEEPSEEK_API_KEY")
)