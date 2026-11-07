"""Configuration constants for Smart Expense Analyzer.

Defines categories, model parameters, thresholds, vocabularies, and UI settings
as specified in the Antigravity build guide and report brief.
"""

from typing import List, Dict

# Application Metadata
APP_NAME = "Smart Expense Analyzer"
APP_CAPTION = (
    "An Explainable Machine-Learning System for Personal Transaction Categorisation "
    "and Spending Anomaly Detection. All data is processed locally in your browser."
)

# Core Categories (exactly eight)
CATEGORIES: List[str] = [
    "Food",
    "Transport",
    "Shopping",
    "Bills",
    "Entertainment",
    "Health",
    "Education",
    "Rent",
]

UNCATEGORISED_LABEL = "Uncategorised"

# File Upload Constraints
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_FILE_EXTENSIONS = [".csv"]
REQUIRED_COLUMNS = ["date", "description", "amount"]

# Classifier Configuration
RANDOM_SEED = 7
CONFIDENCE_THRESHOLD = 0.35
SAMPLES_PER_CATEGORY = 120  # Total = 8 * 120 = 960 rows
LOGISTIC_REGRESSION_C = 8.0
LOGISTIC_REGRESSION_MAX_ITER = 1000

# Feature Extraction Settings
WORD_TFIDF_NGRAM_RANGE = (1, 2)
CHAR_TFIDF_NGRAM_RANGE = (3, 5)

# Anomaly Detection Parameters
MIN_ANOMALY_HISTORY = 5
ANOMALY_MODIFIED_Z_THRESHOLD = 3.5
MODIFIED_Z_MULTIPLIER = 0.6745
MAD_FLOOR_FACTOR = 0.05
MAD_FLOOR_EPSILON = 1e-9

# Duplicate Detection Parameters
DUPLICATE_MAX_DAYS_GAP = 2

# Reporting & Forecasting Parameters
FORECAST_MOVING_AVERAGE_MONTHS = 3
CURRENCY_SYMBOL = "Rs."

# Synthetic Merchant Seed Vocabulary (10 merchants per category)
SEED_MERCHANTS: Dict[str, List[str]] = {
    "Food": [
        "Swiggy",
        "Zomato",
        "McDonalds",
        "Starbucks Coffee",
        "Dominos Pizza",
        "Chai Point",
        "Haldirams Sweets",
        "Subway Restaurant",
        "KFC Fried Chicken",
        "Burger King",
    ],
    "Transport": [
        "Uber India",
        "Ola Cabs",
        "Metro Rail Station",
        "IRCTC Rail Booking",
        "IOCL Petrol Pump",
        "HPCL Fuel Station",
        "Shell Auto Fuel",
        "Rapido Bike Taxi",
        "Fastag Toll Plaza",
        "Auto Rickshaw UPI",
    ],
    "Shopping": [
        "Amazon India",
        "Flipkart Internet",
        "Myntra Fashion",
        "Zara Clothing",
        "HM Retail",
        "Reliance Digital",
        "DMart Supermarket",
        "Croma Electronics",
        "Decathlon Sports",
        "Nykaa Beauty",
    ],
    "Bills": [
        "BESCOM Electricity",
        "Airtel Postpaid Bill",
        "Jio Fiber Broadband",
        "Tata Power Utility",
        "Water Supply Board",
        "Mahanagar Gas Limited",
        "ACT Fibernet",
        "Electricity Bill Pay",
        "Torrent Power",
        "HP Gas Cylinder",
    ],
    "Entertainment": [
        "Netflix Streaming",
        "Spotify Subscription",
        "BookMyShow Tickets",
        "PVR Cinemas",
        "Disney Hotstar",
        "Amazon Prime Video",
        "SonyLIV Entertainment",
        "Smaaash Gaming Arcade",
        "Bowling Alley Lounge",
        "Steam Games Purchase",
    ],
    "Health": [
        "Apollo Pharmacy",
        "Practo Consultation",
        "Dr Lal Pathlabs",
        "Max Healthcare",
        "Medplus Chemist",
        "Care Hospital Clinic",
        "Dental Care Clinic",
        "1mg Health Store",
        "Metropolis Diagnostics",
        "Fortis Hospital OPD",
    ],
    "Education": [
        "Coursera Learning",
        "Udemy Online Course",
        "University Tuition Fee",
        "College Bookstore",
        "Edx Certificate",
        "Oxford Book Store",
        "Exam Registration Fee",
        "Tuition Center Classes",
        "Skillshare Annual",
        "Coaching Institute",
    ],
    "Rent": [
        "Apartment Monthly Rent",
        "House Rent Landlord",
        "PG Accommodation Rent",
        "Flat Society Maintenance",
        "Landlord Transfer Rent",
        "Society Maintenance Charges",
        "Room Rent Payment",
        "Hostel Fee Monthly",
        "Co-Living Monthly Rent",
        "Property Lease Rent",
    ],
}

# Payment channel and location noise tokens
PAYMENT_CHANNELS: List[str] = [
    "UPI",
    "POS",
    "NETBANKING",
    "NEFT",
    "IMPS",
    "RTGS",
    "DEBIT CARD",
    "GPAY",
    "PAYTM",
    "PHONEPE",
    "CREDIT CARD",
    "QR CODE",
]

CITIES: List[str] = [
    "Mumbai",
    "Bengaluru",
    "Delhi",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata",
    "Ahmedabad",
    "Jaipur",
    "Kochi",
]

