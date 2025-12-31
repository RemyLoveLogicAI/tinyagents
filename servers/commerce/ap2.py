from typing import Dict, Any, List
import random
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("commerce")

# Mock Database simulating "Google Shopping Graph" (50B+ listings mocked)
PRODUCTS = [
    {"id": "prod_001", "name": "QuantumX Smartwatch", "category": "Electronics", "list_price": 299.99, "min_price": 250.00, "stock": 50},
    {"id": "prod_002", "name": "NeuroLink Headset", "category": "Electronics", "list_price": 599.99, "min_price": 520.00, "stock": 15},
    {"id": "prod_003", "name": "EcoFresh Air Purifier", "category": "Home", "list_price": 149.99, "min_price": 120.00, "stock": 100},
    {"id": "prod_004", "name": "Titanium Travel Mug", "category": "Accessories", "list_price": 45.00, "min_price": 35.00, "stock": 200},
]

# In-memory storage for active negotiations
negotiations: Dict[str, Any] = {}

@mcp.tool()
async def search_products(query: str) -> str:
    """Search for products in the Google Shopping Graph.

    Args:
        query: Search keywords or intent (e.g., "running shoes under $100").
    """
    query = query.lower()
    results = []
    for p in PRODUCTS:
        if query in p["name"].lower() or query in p["category"].lower():
            results.append(p)

    if not results:
        return "No products found matching your criteria."

    response = "Found the following products in the Global Shopping Graph:\n"
    for p in results:
        response += f"- [{p['id']}] {p['name']} (List Price: ${p['list_price']})\n"
    return response

@mcp.tool()
async def negotiate_price(product_id: str, offer_price: float) -> str:
    """Initiate or continue a negotiation for a product using a Price Mandate.

    Args:
        product_id: The ID of the product.
        offer_price: The user's target price (Price Mandate).
    """
    product = next((p for p in PRODUCTS if p["id"] == product_id), None)
    if not product:
        return f"Product {product_id} not found."

    if product["stock"] <= 0:
        return f"Product {product_id} is out of stock."

    # Simple negotiation logic
    min_price = product["min_price"]
    list_price = product["list_price"]

    if offer_price >= list_price:
        negotiations[product_id] = offer_price
        return f"Offer accepted immediately! Price: ${offer_price}. You can proceed to checkout."

    if offer_price < min_price:
        counter_offer = min_price + (list_price - min_price) * 0.5
        return f"Offer of ${offer_price} is too low. The merchant counters with ${counter_offer:.2f}."

    # Accept if within range
    negotiations[product_id] = offer_price
    return f"Negotiation successful! The merchant has accepted your Price Mandate of ${offer_price}. Proceed to checkout."

@mcp.tool()
async def checkout(product_id: str, confirmed_price: float, payment_method: str = "Google Pay") -> str:
    """Execute the transaction via Agentic Checkout.

    Args:
        product_id: The ID of the product to purchase.
        confirmed_price: The agreed-upon price.
        payment_method: Payment method to use (default: Google Pay).
    """
    product = next((p for p in PRODUCTS if p["id"] == product_id), None)
    if not product:
        return "Transaction failed: Product not found."

    # verify negotiation
    if product_id not in negotiations:
        # If paying full list price, it's allowed without prior negotiation step
        if confirmed_price < product["list_price"]:
             return "Transaction failed: Price not authorized. Please negotiate first."

    # In a real system, we'd verify the price matches the negotiated price exactly
    # For this mock, we assume the agent passes the correct negotiated price.

    product["stock"] -= 1
    return f"""
Transaction Executed Successfully!
----------------------------------
Product: {product['name']}
Price: ${confirmed_price:.2f}
Payment: {payment_method}
Protocol: Google AP2 Agentic Checkout
Status: CONFIRMED
Remaining Stock: {product['stock']}
"""

if __name__ == "__main__":
    mcp.run(transport="stdio")
