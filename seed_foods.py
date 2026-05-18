"""
Day 5: Seed the `foods` table with Voyage AI embeddings.

Run once after creating the table:
    python seed_foods.py

Uses Voyage's voyage-3-lite (512-dim). Batches all foods in one API call
because Voyage accepts up to 128 texts per request.

input_type="document" — tells Voyage these texts will be STORED for
retrieval. The model produces vectors tuned for that role.
"""

import os
import voyageai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)
voyage = voyageai.Client(api_key=os.environ["VOYAGEAI_API_KEY"])

EMBED_MODEL = "voyage-3-lite"

FOODS = [
    # --- Breakfast / breads ---
    {"name": "Oatmeal (1 cup cooked)", "calories": 150, "carbs": 27, "protein": 5, "fat": 3,
     "description": "Oatmeal porridge cooked oats breakfast cereal whole grain healthy fiber morning"},
    {"name": "Eggs (2 large boiled)", "calories": 140, "carbs": 1, "protein": 12, "fat": 10,
     "description": "Eggs boiled hard soft poached protein breakfast chicken egg high protein"},
    {"name": "Scrambled Eggs (2)", "calories": 180, "carbs": 1, "protein": 12, "fat": 14,
     "description": "Scrambled eggs cooked butter breakfast protein"},
    {"name": "Idli (2 pieces)", "calories": 120, "carbs": 26, "protein": 4, "fat": 0.5,
     "description": "Idli steamed rice cake South Indian breakfast fermented urad dal sambar"},
    {"name": "Dosa (plain)", "calories": 170, "carbs": 30, "protein": 4, "fat": 4,
     "description": "Dosa South Indian crepe rice lentil fermented pancake breakfast"},
    {"name": "Roti (1 medium)", "calories": 120, "carbs": 18, "protein": 3, "fat": 3.7,
     "description": "Roti chapati phulka Indian flatbread whole wheat unleavened bread tortilla atta"},
    {"name": "Paratha (plain, 1)", "calories": 260, "carbs": 36, "protein": 6, "fat": 10,
     "description": "Paratha Indian flatbread pan-fried layered ghee oil whole wheat"},
    {"name": "Whole Wheat Bread (1 slice)", "calories": 80, "carbs": 14, "protein": 4, "fat": 1,
     "description": "Bread whole wheat sliced toast sandwich breakfast"},
    {"name": "Poha (1 bowl)", "calories": 250, "carbs": 45, "protein": 5, "fat": 5,
     "description": "Poha flattened rice Indian breakfast snack onion peanut light"},
    {"name": "Upma (1 bowl)", "calories": 230, "carbs": 38, "protein": 6, "fat": 6,
     "description": "Upma South Indian breakfast semolina rava savory porridge"},

    # --- Mains / curries / proteins ---
    {"name": "Dal (1 bowl)", "calories": 180, "carbs": 25, "protein": 12, "fat": 4,
     "description": "Dal lentil curry tadka yellow toor masoor moong Indian protein vegetarian"},
    {"name": "Rajma (1 bowl)", "calories": 220, "carbs": 32, "protein": 14, "fat": 4,
     "description": "Rajma kidney bean curry North Indian Punjabi vegetarian protein"},
    {"name": "Chicken Curry (1 bowl)", "calories": 320, "carbs": 8, "protein": 28, "fat": 20,
     "description": "Chicken curry Indian masala gravy protein non-vegetarian"},
    {"name": "Chicken Breast (100g grilled)", "calories": 165, "carbs": 0, "protein": 31, "fat": 3.6,
     "description": "Chicken breast grilled baked skinless lean protein high protein low fat"},
    {"name": "Paneer Butter Masala (1 bowl)", "calories": 380, "carbs": 12, "protein": 18, "fat": 28,
     "description": "Paneer butter masala makhani Indian creamy tomato curry vegetarian cottage cheese"},
    {"name": "Paneer Tikka (4 pieces)", "calories": 250, "carbs": 6, "protein": 18, "fat": 17,
     "description": "Paneer tikka grilled cottage cheese tandoor Indian appetizer vegetarian"},
    {"name": "Salmon (100g grilled)", "calories": 208, "carbs": 0, "protein": 20, "fat": 13,
     "description": "Salmon fish grilled baked omega-3 healthy fats lean protein seafood"},
    {"name": "Egg Curry (2 eggs)", "calories": 270, "carbs": 8, "protein": 16, "fat": 18,
     "description": "Egg curry Indian masala gravy protein"},
    {"name": "Mixed Vegetable Curry", "calories": 180, "carbs": 20, "protein": 5, "fat": 9,
     "description": "Mixed vegetable curry sabzi Indian vegetarian sabji"},
    {"name": "Aloo Gobi", "calories": 170, "carbs": 22, "protein": 4, "fat": 8,
     "description": "Aloo gobi potato cauliflower Indian dry curry vegetarian"},
    {"name": "Palak Paneer", "calories": 280, "carbs": 12, "protein": 14, "fat": 20,
     "description": "Palak paneer spinach cottage cheese Indian curry vegetarian iron green"},

    # --- Rice / grains ---
    {"name": "Brown Rice (1 cup cooked)", "calories": 215, "carbs": 45, "protein": 5, "fat": 1.8,
     "description": "Brown rice whole grain cooked side dish carbs healthy"},
    {"name": "White Rice (1 cup cooked)", "calories": 200, "carbs": 45, "protein": 4, "fat": 0.5,
     "description": "White rice basmati cooked side dish carbs Indian Asian"},
    {"name": "Biryani (1 cup)", "calories": 290, "carbs": 38, "protein": 12, "fat": 10,
     "description": "Biryani rice meat vegetable Indian Hyderabadi spiced cardamom basmati"},
    {"name": "Quinoa (1 cup cooked)", "calories": 220, "carbs": 39, "protein": 8, "fat": 4,
     "description": "Quinoa whole grain protein gluten free cooked side healthy"},

    # --- Fruits / vegetables ---
    {"name": "Apple (medium)", "calories": 95, "carbs": 25, "protein": 0.5, "fat": 0.3,
     "description": "Apple fruit fresh raw fiber healthy snack"},
    {"name": "Banana (medium)", "calories": 105, "carbs": 27, "protein": 1, "fat": 0,
     "description": "Banana fruit potassium ripe quick energy pre-workout"},
    {"name": "Avocado (half)", "calories": 160, "carbs": 9, "protein": 2, "fat": 15,
     "description": "Avocado healthy fats monounsaturated fruit guacamole creamy"},
    {"name": "Mixed Green Salad", "calories": 50, "carbs": 10, "protein": 2, "fat": 0,
     "description": "Salad lettuce spinach greens vegetables fresh low calorie"},
    {"name": "Orange (medium)", "calories": 60, "carbs": 15, "protein": 1, "fat": 0,
     "description": "Orange fruit citrus vitamin C fresh sweet"},

    # --- Snacks / drinks ---
    {"name": "Greek Yogurt (1 cup)", "calories": 100, "carbs": 6, "protein": 17, "fat": 0.5,
     "description": "Greek yogurt high protein probiotic curd low fat dairy"},
    {"name": "Almonds (30g)", "calories": 170, "carbs": 6, "protein": 6, "fat": 15,
     "description": "Almonds nuts healthy fats snack protein vitamin E"},
    {"name": "Peanut Butter (2 tbsp)", "calories": 190, "carbs": 8, "protein": 8, "fat": 16,
     "description": "Peanut butter spread protein fat snack toast"},
    {"name": "Protein Shake (1 scoop)", "calories": 120, "carbs": 5, "protein": 24, "fat": 1,
     "description": "Protein shake whey powder supplement post workout muscle"},
    {"name": "Coffee with milk", "calories": 60, "carbs": 6, "protein": 3, "fat": 2.5,
     "description": "Coffee latte cappuccino milk hot beverage morning caffeine"},
    {"name": "Chai (1 cup)", "calories": 80, "carbs": 12, "protein": 2, "fat": 3,
     "description": "Chai masala tea Indian milk sugar spiced cardamom ginger"},
    {"name": "Green Tea (1 cup)", "calories": 2, "carbs": 0, "protein": 0, "fat": 0,
     "description": "Green tea matcha antioxidant zero calorie beverage healthy"},
    {"name": "Dark Chocolate (30g)", "calories": 170, "carbs": 13, "protein": 2, "fat": 12,
     "description": "Dark chocolate cocoa cacao snack dessert antioxidants"},

    # --- Junk / fast food (for honest tracking) ---
    {"name": "Pizza Slice (cheese)", "calories": 285, "carbs": 36, "protein": 12, "fat": 10,
     "description": "Pizza slice cheese fast food Italian dough cheese tomato"},
    {"name": "French Fries (medium)", "calories": 365, "carbs": 48, "protein": 4, "fat": 17,
     "description": "French fries chips fried potato fast food snack"},
    {"name": "Samosa (1 piece)", "calories": 260, "carbs": 30, "protein": 5, "fat": 14,
     "description": "Samosa Indian snack deep fried potato pastry triangle"},
    {"name": "Pakora (4 pieces)", "calories": 230, "carbs": 20, "protein": 6, "fat": 14,
     "description": "Pakora bhajji Indian fritter deep fried gram flour besan snack"},
    {"name": "Burger (beef)", "calories": 540, "carbs": 40, "protein": 25, "fat": 30,
     "description": "Burger beef hamburger fast food bun patty cheese lettuce"},
    {"name": "Ice Cream (1 scoop)", "calories": 200, "carbs": 24, "protein": 3, "fat": 11,
     "description": "Ice cream dessert frozen sweet vanilla chocolate"},
    {"name": "Coca-Cola (330ml can)", "calories": 140, "carbs": 39, "protein": 0, "fat": 0,
     "description": "Coca cola Coke soda fizzy drink sugar carbonated beverage"},
]


def main():
    # Wipe existing rows so re-running is idempotent
    supabase.table("foods").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

    descriptions = [f["description"] for f in FOODS]

    # Batch all descriptions in a single API call — Voyage allows up to 128.
    print(f"Embedding {len(descriptions)} foods in one batch…")
    result = voyage.embed(
        texts=descriptions,
        model=EMBED_MODEL,
        input_type="document",      # docs are stored for retrieval
    )

    for food, embedding in zip(FOODS, result.embeddings):
        food["embedding"] = embedding
        supabase.table("foods").insert(food).execute()
        print(f"  inserted: {food['name']}")

    print(f"\nDone — {len(FOODS)} foods seeded.")
    print(f"Total tokens used: {result.total_tokens}")


if __name__ == "__main__":
    main()