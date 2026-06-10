import time
import random
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

os.makedirs("data/raw", exist_ok=True)

PRODUCTS = [
    ("https://www.bodybuilding.com/store/opt/whey.html", "protein"),
    ("https://www.bodybuilding.com/store/opt/creatine.html", "creatine"),
]

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def scrape_amazon_reviews():
    driver = setup_driver()
    reviews = []
    
    urls = [
        ("https://www.amazon.com/Optimum-Nutrition-Standard-Protein-Chocolate/product-reviews/B000QSNYGI", "protein"),
        ("https://www.amazon.com/Optimum-Nutrition-Micronized-Creatine-Monohydrate/product-reviews/B002E7GIZK", "creatine"),
        ("https://www.amazon.com/Cellucor-C4-Original-Pre-Workout-Watermelon/product-reviews/B00J5MN9QE", "pre-workout"),
    ]
    
    for url, category in urls:
        print(f"Scraping {category} reviews...")
        try:
            driver.get(url)
            time.sleep(random.uniform(3, 5))
            
            review_elements = driver.find_elements(By.CSS_SELECTOR, "[data-hook='review-body'] span")
            rating_elements = driver.find_elements(By.CSS_SELECTOR, "[data-hook='review-star-rating'] span")
            
            print(f"Found {len(review_elements)} reviews")
            
            for i, elem in enumerate(review_elements[:20]):
                text = elem.text.strip()
                if len(text) > 20:
                    rating = 5
                    if i < len(rating_elements):
                        rating_text = rating_elements[i].get_attribute("innerHTML")
                        try:
                            rating = int(rating_text[0])
                        except:
                            rating = 5
                    
                    if rating >= 4:
                        sentiment = "positive"
                        label = 2
                    elif rating <= 2:
                        sentiment = "negative"
                        label = 0
                    else:
                        sentiment = "neutral"
                        label = 1
                    
                    reviews.append({
                        "text": text,
                        "rating": rating,
                        "sentiment": sentiment,
                        "label": label,
                        "category": category
                    })
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(random.uniform(3, 5))
    
    driver.quit()
    return reviews

reviews = scrape_amazon_reviews()
print(f"\nTotal: {len(reviews)} reviews")

if reviews:
    df = pd.DataFrame(reviews)
    df.to_csv("data/raw/fitness_reviews.csv", index=False)
    print(df["sentiment"].value_counts())
    print("\nSample:", reviews[0]["text"][:100])
else:
    print("No reviews scraped")
