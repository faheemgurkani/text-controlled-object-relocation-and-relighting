from openai import OpenAI
import json



# def parse_instruction(instruction, api_key):
#     client = OpenAI(api_key=api_key)

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "Extract object, action, location, lighting from input."},
#             {"role": "user", "content": instruction}
#         ],
#     )
    
#     content = response.choices[0].message.content
    
#     return json.loads(content)

def parse_instruction(user_instruction, system_instruction, api_key):
    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction}
        ],
    )

    # print(resp) # For, testing

    content = resp.choices[0].message.content.strip()
    
    if not content:
        raise ValueError("Empty response from OpenAI — no content to parse!")
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Raw content:", repr(content))
        
        raise ValueError("Failed to parse JSON from OpenAI response.")