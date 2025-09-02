from openai import OpenAI
from google import generativeai as genai
from mistralai import Mistral
import anthropic

import pandas as pd


def call_llm_api(basepath, prompt, model):
    if model == 'dummy':
        # Dummy model for testing purposes
        #print("Using dummy model. Returning a dummy response.")
        return "This is a dummy response."

    elif model == 'gpt-3.5-turbo':
        # Call OpenAI GPT-3.5 Turbo API

        keys_df = pd.read_csv(f'{basepath}/secrets/api_keys.csv')
        fildf = keys_df[keys_df['provider'] == 'openai']
        api_key = fildf['key'].values[0]
        client = OpenAI(
            api_key=api_key
        )

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        i = 0
        # Print the response from the model

        #print("Response with GPT-3.5 Turbo API:")
        for choice in completion.choices:
            #print(f'Choice {i + 1}\n', choice.message.content)
            i += 1

        return completion.choices[0].message.content

    elif model == 'gpt-4':
        # Call OpenAI GPT-4 API

        keys_df = pd.read_csv(f'{basepath}/secrets/api_keys.csv')
        fildf = keys_df[keys_df['provider'] == 'openai']
        api_key = fildf['key'].values[0]
        client = OpenAI(
            api_key=api_key
        )

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        i = 0
        #print("Response with GPT-4 API:")
        for choice in completion.choices:
            #print(f'Choice {i + 1}\n', choice.message.content)
            i += 1

        return completion.choices[0].message.content

    elif model == 'gemini-2.0-flash-001':
        # Call Google Gemini API
        # Note: Replace with actual API call for Gemini
        keys_df = pd.read_csv(f'{basepath}/secrets/api_keys.csv')
        fildf = keys_df[keys_df['provider'] == 'gemini']
        api_key = fildf['key'].values[0]

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")

        response = model.generate_content(prompt)
        #print("Response with Gemini API:")
        #print(response.text)
        return response.text

    elif model == "deepseek-reasoner":
        # Call DeepSeek Reasoner API
        # Note: Replace with actual API call for DeepSeek
        keys_df = pd.read_csv(f'{basepath}/secrets/api_keys.csv')
        fildf = keys_df[keys_df['provider'] == 'deepseek']
        api_key = fildf['key'].values[0]

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # Round 1
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )

        reasoning_content = response.choices[0].message.reasoning_content
        return_val = response.choices[0].message.content

        #print("Response with DeepSeek Reasoner API:")
        #print("Reasoning Content:", reasoning_content)
        #print("Final Answer:", return_val)

        return return_val

    elif model == "deepseek-chat":
        # Call DeepSeek Reasoner API
        # Note: Replace with actual API call for DeepSeek
        keys_df = pd.read_csv(f'{basepath}/secrets/api_keys.csv')
        fildf = keys_df[keys_df['provider'] == 'deepseek']
        api_key = fildf['key'].values[0]

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # Round 1
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )

        # reasoning_content = response.choices[0].message.reasoning_content
        return_val = response.choices[0].message.content

        #print("Response with DeepSeek V3 API:")
        # print("Reasoning Content:", reasoning_content)
        #print("Final Answer:", return_val)

        return return_val

    elif model == "mistral-large-2411":
        # Call Mistral AI API

        keys_df = pd.read_csv(f'{basepath}/secrets/api_keys.csv')
        fildf = keys_df[keys_df['provider'] == 'mistral']
        api_key = fildf['key'].values[0]

        model = "mistral-large-2411"

        client = Mistral(api_key=api_key)

        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        #print("Response with Mistral AI API:")
        #print(chat_response.choices[0].message.content)
        return chat_response.choices[0].message.content


    elif model == "claude-opus-4-20250514":
        # Call Anthropic Claude API
        # Note: Replace with actual API call for Claude
        keys_df = pd.read_csv(f'{basepath}/secrets/api_keys.csv')
        fildf = keys_df[keys_df['provider'] == 'claude']
        api_key = fildf['key'].values[0]
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-20250514",  # or other Claude model
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        #print("Response with Claude Opus 4 API:")
        #print(message.content[0].text)
        return message.content[0].text

    elif model == "llama-3.1":
        # Call Llama 3.1 API
        # Note: Replace with actual API call for Llama
        response = "llama-3.1 is not supported in this version of the code. "
        return response

    elif model == "grok-3":
        keys_df = pd.read_csv(f'{basepath}/secrets/api_keys.csv')
        fildf = keys_df[keys_df['provider'] == 'grok']
        api_key = fildf['key'].values[0]
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

        completion = client.chat.completions.create(
            model="grok-3",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        i = 0
        #print("Response with Grok-3 API:")
        for choice in completion.choices:
            #print(f'Choice {i + 1}\n', choice.message.content)
            i += 1

        return completion.choices[0].message.content


def print_output_for_all(basepath, zero_shot_prompt):
    prediction_gpt_3_5 = call_llm_api(basepath, zero_shot_prompt, model='gpt-3.5-turbo')

    prediction_gpt_4 = call_llm_api(basepath, zero_shot_prompt, model='gpt-4')

    prediction_gemini =  call_llm_api(basepath, zero_shot_prompt, model='gemini-2.0-flash-001')

    # prediction_deepseek = call_llm_api(basepath, zero_shot_prompt, model='deepseek-reasoner')

    prediction_deepseek =  call_llm_api(basepath, zero_shot_prompt, model='deepseek-chat')

    prediction_mistral =  call_llm_api(basepath, zero_shot_prompt, model='mistral-large-2411')

    prediction_claude = call_llm_api(basepath, zero_shot_prompt, model='claude-opus-4-20250514')

    #prediction_llama =  call_llm_api(basepath, zero_shot_prompt, model='llama-3.1')

    prediction_grok =  call_llm_api(basepath, zero_shot_prompt, model='grok-3')

    # return a dictionary with all predictions
    return_dict = {"gpt-3.5-turbo": prediction_gpt_3_5,
                    "gpt-4": prediction_gpt_4,
                    "gemini-2.0-flash-001": prediction_gemini,
                    "deepseek-chat": prediction_deepseek,
                    "mistral-large-2411": prediction_mistral,
                    "claude-opus-4-20250514": prediction_claude,
                    #"llama-3.1": prediction_llama,
                    "grok-3": prediction_grok}
    return return_dict