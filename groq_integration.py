import os
from dotenv import load_dotenv
import time
from groq import Groq





# Load environment variables
load_dotenv()

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = 'llama-3.2-90b-text-preview'





def identify_query_category(query):
    # Refined prompt to classify the query into one of the six categories with brief descriptions for each
    refined_query = (
        f"Classify the following query into one of these categories based on its content: "
        f"[Address] for street addresses or specific locations, "
        f"[Category] for general types of places like parks or hospitals, including queries with 'nearby' or 'close to current location',"
        f"[Brand] for queries mentioning specific brand names like Starbucks, McDonald's, etc., "
        f"[Quantity] for queries specifying any number of places. Any input query where the user has specified the quantity of places then it comes under this category even if it is a brand, "
        f"[Distance] for queries mentioning any specific distance (e.g., 1 km), which should take precedence over any other category, and"
        f"[Location with Reference] only for queries that mention two distinct locations, like 'near Amsterdam Central Station' or 'close to Vondelpark'."
        f"Ensure that only one category is returned at a time, based on the content. "
        f"Return only the category name in brackets and nothing else. "
        f"Query: '{query}'"
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": refined_query}
        ]
    }

    start_time = time.time()
    response = client.chat.completions.create(messages=body['messages'], model=MODEL)
    end_time = time.time()
    response_time = end_time - start_time

    print(f"Response Time: {response_time} seconds")

    # Return the category extracted from the response
    return response.choices[0].message.content.strip()





# Function to Handle [Address] category
def process_address_query(query):
    # Refined prompt to handle the address query
    address_prompt = (
        f"Correct any spelling mistakes, extract the address from the following query, "
        f"and format it into this universal address format: [Street Address, Locality (if applicable), "
        f"City, Administrative Area (if applicable), Postal Code, Country]. "
        f"Do not include any speculative information, explanations, or assumptions. "
        f"Return only the formatted address in square brackets inside [] and nothing else. "
        f"Query: '{query}'"
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": address_prompt}
        ]
    }

    start_time = time.time()
    response = client.chat.completions.create(messages=body['messages'], model=MODEL)
    end_time = time.time()
    response_time = end_time - start_time

    print(f"Response Time: {response_time} seconds")

    # Return the formatted address extracted from the response
    return response.choices[0].message.content.strip()





# Function to handle [Category] queries
def process_category_query(query):
    category_prompt = (
        f"Analyze the following query to extract the type of place where the user intends to go. "
        f"Correct any spelling mistakes if necessary, but do not include any commentary or corrections in the response. "
        f"Return only the most relevant place in a maximum of 1 or 2 words in square brackets, appropriate for sending to geocoder. "
        f"Only provide the place name in square brackets, without any explanations, assumptions, or unnecessary information. "
        f"Query: '{query}'"
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": category_prompt}
        ]
    }

    start_time = time.time()
    response = client.chat.completions.create(messages=body['messages'], model=MODEL)
    end_time = time.time()
    response_time = end_time - start_time

    print(f"Response Time: {response_time} seconds")
    return response.choices[0].message.content.strip()





# Function to handle [Quantity] queries
def process_quantity_query(query):
    quantity_prompt = (
        f"Analyze the query to extract both the type of place where the user intends to go and the number of places the user is requesting. "
        f"Correct any spelling mistakes if necessary "
        f"Return only the most relevant place in a maximum of 1 or 2 words, appropriate for sending to geocoder. "
        f"Return the output in this format: [(Name of Place), (Quantity)]. For example, if the user is asking for 3 parks, return '[(Park), (3)]'. "
        f"Only provide the name of the place and the quantity in square brackets, with no additional information in the response. "
        f"Query: '{query}'"
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": quantity_prompt}
        ]
    }

    start_time = time.time()
    response = client.chat.completions.create(messages=body['messages'], model=MODEL)
    end_time = time.time()
    response_time = end_time - start_time

    print(f"Response Time: {response_time} seconds")
    return response.choices[0].message.content.strip()





# Function to handle [Brand] queries
def process_brand_query(query):
    brand_prompt = (
        f"Analyze the following query to extract the specific brand name where the user intends to go. "
        f"Correct any spelling mistakes if necessary, but do not include any commentary or corrections in the response. "
        f"Return only the brand name in square brackets, appropriate for sending to a geocoder. "
        f"Sometimes some brands are into multiple business, so if the query has mentioned about the intended place to go for that brand, only in that case mention the brand name followed by the intended place to go"
        f"Only provide the brand name in square brackets, with no additional information. "
        f"Query: '{query}'"
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": brand_prompt}
        ]
    }

    start_time = time.time()
    response = client.chat.completions.create(messages=body['messages'], model=MODEL)
    end_time = time.time()
    response_time = end_time - start_time

    print(f"Response Time: {response_time} seconds")
    return response.choices[0].message.content.strip()





# Function to handle [Distance] queries
def process_distance_query(query):
    distance_prompt = (
        f"Analyze the query to extract both the type of place where the user intends to go in the sense that the place name can be sent to a geocoder, and the distance, including the unit. "
        f"Correct any spelling mistakes if necessary but do not include any commentary or corrections in the response. "
        f"Return the output in this format: [(Name of Place), (Distance with Unit)]. "
        f"For example, if the user is asking for restaurants within 10 km, return '[(Restaurant), (10 km)]'. "
        f"Only provide the name of the place and the distance in square brackets, with no additional information in the response. "
        f"Query: '{query}'"
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": distance_prompt}
        ]
    }

    start_time = time.time()
    response = client.chat.completions.create(messages=body['messages'], model=MODEL)
    end_time = time.time()
    response_time = end_time - start_time

    print(f"Response Time: {response_time} seconds")
    return response.choices[0].message.content.strip()





# Function to handle [Location with Reference] queries
def process_location_with_reference_query(query):
    reference_prompt = (
        f"Analyze the query to extract two locations: the starting location (where the user intends to go) and the reference location. "
        f"Correct any spelling mistakes if necessary but do not include any commentary or corrections in the response. "
        f"Return the output in this format: [(Start Location), (End Location)]. "
        f"For example, if the user is asking for hotels near Schiphol Airport, return '[(Hotel), (Schiphol Airport)]'. "
        f"Only provide the start location and reference location in square brackets, with no additional information in the response. "
        f"Query: '{query}'"
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": reference_prompt}
        ]
    }

    start_time = time.time()
    response = client.chat.completions.create(messages=body['messages'], model=MODEL)
    end_time = time.time()
    response_time = end_time - start_time

    print(f"Response Time: {response_time} seconds")
    return response.choices[0].message.content.strip()





# Function to process queries based on the identified category
def process_query_by_category(query, category):
    if category == "[Address]":
        return process_address_query(query)
    elif category == "[Category]":
        return process_category_query(query)
    elif category == "[Brand]":
        return process_brand_query(query)
    elif category == "[Quantity]":
        return process_quantity_query(query)
    elif category == "[Distance]":
        return process_distance_query(query)
    elif category == "[Location with Reference]":
        return process_location_with_reference_query(query)
    else:
        return "Query does not belong to the recognized categories."





# # Example query for testing
# # user_query = "I want to go to kattenburgerstraat 5"
# user_query = "Find a shopping mall close to Amsterdam Arena."
# print(f"User Query: {user_query}")
#
# # Identify the category of the query
# category = identify_query_category(user_query)
# print(f"Query Category: {category}")
#
# # Process the query based on the identified category using the new function
# processed_output = process_query_by_category(user_query, category)
# print(f"Processed Output: {processed_output}")

# End of Python Script
