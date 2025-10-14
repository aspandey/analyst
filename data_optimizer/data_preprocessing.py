import re
import datasource.parse_excel as pe

def data_normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("&", "and")
    text = re.sub(r"\bltd\.?\b", "limited", text)
    text = re.sub(r"\bpvt\.?\b", "private", text)
    text = re.sub(r"\bco\.?\b", "company", text)
    text = re.sub(r"\bcorp\.?\b", "corporation", text)
    text = re.sub(r"\binc\.?\b", "incorporated", text)
    text = re.sub(r"\bplc\b", "public limited company", text)
    text = re.sub(r"\bgroup\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation except spaces
    text = re.sub(r'\s+', ' ', text)  # Clean up spaces again
    return text.strip()


SECTOR_MAPPING = {
    "banks": "banking",
    "finance": "financial services",
    "auto": "automobile",
    "automobiles": "automobile",
    "it": "information technology",
    "pharma": "pharmaceuticals",
    "healthcare": "healthcare services",
    "retailing": "retail",
    # Add more mappings as needed
}

def data_normalize_sector(sector: str) -> str:
    if not sector:
        return ""
    normalized = sector.strip().lower()
    return SECTOR_MAPPING.get(normalized, normalized)


def data_preprocess_stock(stocks_info: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Preprocess and enrich stock data for vector insertion in Weaviate.
    Adds normalized text and a combined_text field.
    """

    processed_data = []

    for item in stocks_info:
        company = data_normalize_text(item.get("company_or_stock_name", ""))
        sector = data_normalize_sector(item.get("industry_sector", ""))
        sector = data_normalize_text(sector)
        pms = data_normalize_text(item.get("portfolio_management_services_name", ""))
        month = data_normalize_text(item.get("data_month", ""))
        
        # Convert numeric values to text for embedding
        qty = item.get("quantity_of_shares", "")
        qty_text = f"{qty} shares" if qty else ""
        
        value = item.get("market_value_lacs_inr", "")
        value_text = f"market value {value} lakh INR" if value else ""
        
        aum = item.get("asset_under_managment_percentage", "")
        aum_text = f"{aum} percent of total assets of pms" if aum else ""
        
        # Create a rich semantic text representation
        combined_text = (
            f"{company} works in {sector} sector. {pms} pms holds {qty_text} of {company} in the month of {month} with {value_text}, \
            representing {aum_text}."
        ).strip()

        processed_data.append({
            "company_or_stock_name": company,
            "industry_sector": sector,
            "quantity_of_shares": item.get("quantity_of_shares"),
            "market_value_lacs_inr": item.get("market_value_lacs_inr"),
            "asset_under_managment_percentage": item.get("asset_under_managment_percentage"),
            "data_month": month,
            "portfolio_management_services_name": pms,
            "combined_text": combined_text
        })

    return processed_data

if __name__ == "__main__":
    stocks_info = pe.get_stock_info_from_xlsx(pe.STOCK_INFO_PATH)
    processed_data = data_preprocess_stock(stocks_info)

    for row in processed_data:
        print(f"\n{row}")