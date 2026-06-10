import pandas as pd
import time
import os
from google_play_scraper import reviews, Sort

os.makedirs("data/raw", exist_ok=True)

APPS = [
    # ── NUTRITION & CALORIE TRACKING ──
    ("com.myfitnesspal.android", "MyFitnessPal", "nutrition"),
    ("com.cronometer.android.gold", "Cronometer", "nutrition"),
    ("com.loseit.loseweight", "Lose It!", "nutrition"),
    ("com.fatsecret.android", "FatSecret", "nutrition"),
    ("com.noom.Noom", "Noom", "nutrition"),
    ("com.sparkpeople.android", "SparkPeople", "nutrition"),
    ("com.lifesum.lifesum", "Lifesum", "nutrition"),
    ("com.yazio.android", "Yazio", "nutrition"),

    # ── SUPPLEMENTS & SPORTS NUTRITION ──
    ("com.myprotein.app", "MyProtein", "supplements"),
    ("com.bodybuilding.store", "Bodybuilding.com", "supplements"),
    ("com.gnc.android", "GNC", "supplements"),
    ("com.optimumnutrition.on", "Optimum Nutrition", "supplements"),
    ("com.iherb.app", "iHerb", "supplements"),
    ("com.vitacost.android", "Vitacost", "supplements"),

    # ── GENERAL WORKOUT & TRAINING ──
    ("com.nike.ntc", "Nike Training Club", "workout"),
    ("air.com.athleanx.athleanx", "AthleanX", "workout"),
    ("com.strongapp.pro", "Strong Workout Tracker", "workout"),
    ("com.jefit.jefit", "JEFIT", "workout"),
    ("com.fitbod.beta", "Fitbod", "workout"),
    ("com.freeletics.freedomapp", "Freeletics", "workout"),
    ("com.dailyburn.android", "DailyBurn", "workout"),
    ("com.google.android.apps.fitness", "Google Fit", "workout"),
    ("com.workout.trainer.workouttrainer", "Workout Trainer", "workout"),
    ("sworkit.android", "Sworkit", "workout"),

    # ── POWERLIFTING & STRENGTH ──
    ("com.stronglifts.app", "StrongLifts 5x5", "powerlifting"),
    ("com.progressionapp", "Progression", "powerlifting"),
    ("com.liftingcast.lifter", "LiftingCast", "powerlifting"),
    ("com.gym.workout.plan.weightlifting", "Gym Workout Plan", "powerlifting"),
    ("com.boostcamp.app", "Boostcamp", "powerlifting"),

    # ── BODYBUILDING SPECIFIC ──
    ("com.arnold.schwarzenegger.fitness", "Arnold Schwarzenegger Fitness", "bodybuilding"),
    ("com.musclemonster.android", "Muscle Monster", "bodybuilding"),
    ("com.gymshark.training", "Gymshark Training", "bodybuilding"),
    ("com.bulkpowders.android", "Bulk Powders", "bodybuilding"),

    # ── CROSSFIT & HIIT ──
    ("com.beyondthewhiteboard.btwb", "BTWB CrossFit", "crossfit"),
    ("com.sugarwod.sugarwod", "SugarWOD", "crossfit"),
    ("com.wodify.android", "Wodify", "crossfit"),
    ("com.7minuteworkout.app", "7 Minute Workout", "hiit"),
    ("com.interval.timer.hiit.tabata", "Interval Timer HIIT", "hiit"),

    # ── RUNNING & CARDIO ──
    ("com.nike.plusgps", "Nike Run Club", "running"),
    ("com.adidas.runtastic", "Runtastic", "running"),
    ("com.strava.android", "Strava", "running"),
    ("com.mapmyfitness.android2", "MapMyFitness", "running"),
    ("com.garmin.android.apps.connectmobile", "Garmin Connect", "running"),
    ("com.runkeeper.android", "RunKeeper", "running"),
    ("com.polar.flow.app", "Polar Flow", "running"),
    ("com.zombiesrungame", "Zombies Run!", "running"),

    # ── CYCLING ──
    ("com.zwift.android.prod", "Zwift", "cycling"),
    ("com.wahoo.fitness", "Wahoo Fitness", "cycling"),
    ("com.trainerroad.android", "TrainerRoad", "cycling"),
    ("com.mapmyride.android2", "MapMyRide", "cycling"),

    # ── SWIMMING ──
    ("com.swimio.app", "Swimio", "swimming"),
    ("com.myswimpro.android", "MySwimPro", "swimming"),

    # ── WEARABLES & RECOVERY ──
    ("com.whoop.android", "Whoop", "recovery"),
    ("com.fitbit.FitbitMobile", "Fitbit", "recovery"),
    ("com.oura.ring2", "Oura Ring", "recovery"),
    ("com.samsung.android.shealth", "Samsung Health", "recovery"),
    ("com.apple.health", "Apple Health", "recovery"),
    ("com.recoveryapp", "Recovery App", "recovery"),

    # ── SLEEP TRACKING ──
    ("com.sleepcycle.android", "Sleep Cycle", "sleep"),
    ("com.urbandroid.sleep", "Sleep as Android", "sleep"),
    ("com.pillow.app", "Pillow Sleep Tracker", "sleep"),
    ("com.nappy.android", "Nappy Sleep", "sleep"),

    # ── GYM & CLASSES ──
    ("com.planetfitness.planetfitnessmobileapp", "Planet Fitness", "gym"),
    ("com.mindbodyonline.connect", "Mindbody", "gym"),
    ("com.peloton.android", "Peloton", "gym"),
    ("com.classpass.android", "ClassPass", "gym"),
    ("com.virtuagym.pro", "VirtuaGym", "gym"),

    # ── YOGA & MINDFULNESS ──
    ("com.glo.app", "Glo Yoga", "yoga"),
    ("com.downdog.app", "Down Dog Yoga", "yoga"),
    ("com.calm.android", "Calm", "yoga"),
    ("com.headspace.android", "Headspace", "yoga"),
    ("com.yogaglo.android", "YogaGlo", "yoga"),
    ("com.prana.breath", "Prana Breath", "yoga"),

    # ── PERSONAL TRAINING & COACHING ──
    ("com.trainerize.android", "Trainerize", "coaching"),
    ("com.mycoach.app", "MyCoach", "coaching"),
    ("com.tonal.android", "Tonal", "coaching"),
    ("com.future.fit", "Future Fitness", "coaching"),
    ("com.vi.trainer", "Vi Trainer", "coaching"),

    # ── WEIGHT LOSS PROGRAMS ──
    ("com.weightwatchers.android", "WeightWatchers", "weight_loss"),
    ("com.noom.Noom", "Noom", "weight_loss"),
    ("com.jenny.craig.android", "Jenny Craig", "weight_loss"),
    ("com.second.nature.app", "Second Nature", "weight_loss"),

    # ── MEAL PREP & DIET ──
    ("com.mealime.android", "Mealime", "meal_prep"),
    ("com.prepear.android", "Prepear", "meal_prep"),
    ("com.eatthismuch.android", "Eat This Much", "meal_prep"),
    ("com.wholefoodmarket.android", "Whole Foods", "meal_prep"),

    # ── PHYSIOTHERAPY & REHAB ──
    ("com.kaia.health", "Kaia Health", "physio"),
    ("com.hinge.health", "Hinge Health", "physio"),
    ("com.curable.app", "Curable", "physio"),
    ("com.recoverapp.android", "Recover Athletics", "physio"),

    # ── MARTIAL ARTS & COMBAT ──
    ("com.graciebarra.app", "Gracie Barra BJJ", "martial_arts"),
    ("com.expertboxing.android", "Expert Boxing", "martial_arts"),
    ("com.mma.training.app", "MMA Training", "martial_arts"),
    ("com.muaythai.app", "Muay Thai", "martial_arts"),

    # ── TEAM SPORTS FITNESS ──
    ("com.nike.sport.running", "Nike Sport", "team_sports"),
    ("com.teamsnap.android", "TeamSnap", "team_sports"),
    ("com.hudl.android", "Hudl", "team_sports"),

    # ── STRETCHING & FLEXIBILITY ──
    ("com.stretching.exercises.flexibility", "Stretching Exercises", "flexibility"),
    ("com.pliability.app", "Pliability", "flexibility"),
    ("com.romwod.android", "ROMWOD", "flexibility"),

    # ── SPORTS TRACKING ──
    ("com.under.armour.record", "Under Armour Record", "sports_tracking"),
    ("com.adidas.running", "Adidas Running", "sports_tracking"),
    ("com.endomondo.android", "Endomondo", "sports_tracking"),
]

all_reviews = []
failed_apps = []

print(f"Starting scrape of {len(APPS)} apps...\n")

for i, (app_id, app_name, category) in enumerate(APPS):
    print(f"[{i+1}/{len(APPS)}] Scraping {app_name} ({category})...")
    try:
        result, _ = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.MOST_RELEVANT,
            count=300
        )

        count = 0
        for r in result:
            text = r['content']
            rating = r['score']

            if not text or len(text) < 15:
                continue

            if rating >= 4:
                sentiment = "positive"
                label = 2
            elif rating <= 2:
                sentiment = "negative"
                label = 0
            else:
                sentiment = "neutral"
                label = 1

            all_reviews.append({
                "text": text[:1000],
                "rating": rating,
                "sentiment": sentiment,
                "label": label,
                "app": app_name,
                "app_id": app_id,
                "category": category
            })
            count += 1

        print(f"  ✅ Got {count} reviews")
        time.sleep(1.5)

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        failed_apps.append(app_name)
        time.sleep(2)

df = pd.DataFrame(all_reviews)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/raw/fitness_reviews.csv", index=False)

print(f"\n{'='*60}")
print(f"SCRAPING COMPLETE!")
print(f"{'='*60}")
print(f"Total reviews:     {len(df)}")
print(f"Total apps:        {df['app'].nunique()}")
print(f"Total categories:  {df['category'].nunique()}")
print(f"Failed apps:       {len(failed_apps)}")
print(f"\nSentiment distribution:")
print(df["sentiment"].value_counts())
print(f"\nCategory breakdown:")
print(df.groupby("category")["text"].count().sort_values(ascending=False))
if failed_apps:
    print(f"\nFailed apps: {failed_apps}")
