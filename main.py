import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from groq_integration import (
    identify_query_category,
    process_address_query,
    process_category_query,
    process_quantity_query,
    process_brand_query,
    process_distance_query,
    process_location_with_reference_query,
    process_query_by_category
)





# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()





# FastAPI GET endpoint to classify the query
@app.get("/classify_query/")
def classify_query(query: str = Query(..., description="Input query to classify the category")):
    """
    Classifies a given natural language query into one of the six categories:
    [Address], [Category], [Brand], [Quantity], [Distance], [Location with Reference].
    """
    category = identify_query_category(query)
    return {"query": query, "category": category}





# FastAPI GET endpoint to process and return a formatted address
@app.get("/classify_query/process_address/")
def process_address(query: str = Query(..., description="Input query to extract and format the address")):
    """
    Extracts the address from a given natural language query, corrects spelling if necessary,
    and returns the address in a universal format: [Street Address, Locality (if applicable),
    City, Administrative Area (if applicable), Postal Code, Country].
    """
    formatted_address = process_address_query(query)
    return {"query": query, "formatted_address": formatted_address}





# FastAPI GET endpoint to process and return a category
@app.get("/classify_query/process_category/")
def process_category(query: str = Query(..., description="Input query to extract and format the category")):
    """
    Extracts the category (type of place) from a given natural language query, corrects spelling if necessary,
    and returns the most relevant place in square brackets.
    """
    formatted_category = process_category_query(query)
    return {"query": query, "formatted_category": formatted_category}





# FastAPI GET endpoint to process and return a quantity
@app.get("/classify_query/process_quantity/")
def process_quantity(query: str = Query(..., description="Input query to extract and format the quantity")):
    """
    Extracts the quantity and the type of place where the user intends to go from a given natural language query.
    Returns the output in the format [(Place), (Quantity)].
    """
    formatted_quantity = process_quantity_query(query)
    return {"query": query, "formatted_quantity": formatted_quantity}





# FastAPI GET endpoint to process and return a brand
@app.get("/classify_query/process_brand/")
def process_brand(query: str = Query(..., description="Input query to extract and format the brand")):
    """
    Extracts the brand name from a given natural language query, corrects spelling if necessary,
    and returns the brand name in square brackets.
    """
    formatted_brand = process_brand_query(query)
    return {"query": query, "formatted_brand": formatted_brand}





# FastAPI GET endpoint to process and return a distance
@app.get("/classify_query/process_distance/")
def process_distance(query: str = Query(..., description="Input query to extract and format the distance")):
    """
    Extracts the type of place and the distance (with unit) from a given natural language query.
    Returns the output in the format [(Place), (Distance with Unit)].
    """
    formatted_distance = process_distance_query(query)
    return {"query": query, "formatted_distance": formatted_distance}





# FastAPI GET endpoint to process and return location with reference
@app.get("/classify_query/process_location_with_reference/")
def process_location_with_reference(query: str = Query(..., description="Input query to extract and format location with reference")):
    """
    Extracts two locations: the starting location and the reference location from a given natural language query.
    Returns the output in the format [(Start Location), (End Location)].
    """
    formatted_reference = process_location_with_reference_query(query)
    return {"query": query, "formatted_reference": formatted_reference}





# FastAPI GET endpoint to process any query based on its identified category
@app.get("/process_query/")
def process_query(query: str = Query(..., description="Input query to process based on the identified category")):
    """
    Classifies the query into one of the six categories and processes it accordingly.
    """
    category = identify_query_category(query)
    processed_output = process_query_by_category(query, category)
    return {"query": query, "category": category, "processed_output": processed_output}

# End of FastAPI Script
