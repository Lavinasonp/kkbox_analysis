<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/d/d4/KKBOX_logo.svg" alt="KKBox Logo" width="300"/>
  <br><br>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"/>
</div>

# KKBox Subscriber Churn Analytics

KKBox is one of Asia's largest music streaming platforms, operating primarily in Taiwan, Hong Kong, and Southeast Asia. This project is a data analytics study of subscriber churn, built on the dataset released by KKBox for the WSDM 2018 challenge on Kaggle.

---

## What problem are we solving

When a subscriber stops renewing their plan, that is churn. For a subscription business like KKBox, losing a subscriber is expensive — not because of the single missed payment, but because of the future revenue that gets cut off. The cost of re-acquiring a churned subscriber through ads or promotions is usually far higher than the cost of retaining them in the first place.

This project tries to understand who is churning, when, and what is driving it. Not through a prediction model, but through a thorough breakdown of the data — looking at how subscribers behave before they leave versus how retained subscribers behave, and finding where the differences are large enough to actually act on.

---

## About the Data

The data covers KKBox subscribers from roughly 2015 to early 2017. The churn label reflects whether a user renewed their subscription in the February–March 2017 window. Listening and transaction history from the months leading up to that window form the behavioural profile used in the analysis.

There are four source tables. The member table holds demographic information — date of birth, gender, city of residence, when they registered, and how they registered (iOS, Android, web browser, etc.). The transactions table has every payment ever made by each subscriber — the plan they bought, the price, how much they actually paid, whether auto-renew was on, and whether they had cancelled at any point. The user logs table has daily listening activity — how many songs were played, how many were completed, how many were skipped. The train table holds the churn label itself, one row per subscriber.

---

## Key Terms Used in This Analysis

**Churn** — A subscriber is marked as churned if they did not make a new payment within 30 days of their last plan expiry. The `is_churn` column is 1 for churned users and 0 for retained users.

**Auto-renew** — A flag on each transaction that indicates whether the subscriber had automatic payment switched on. When it is on, the plan renews without the user needing to do anything. This turned out to be one of the most important signals in the entire dataset.

**Promo / promotional transaction** — A transaction where the user paid significantly less than the listed plan price, or paid nothing at all. Promo ratio is the share of a user's transactions that were promotional. Someone who has consistently paid full price behaves very differently from someone who has mostly been on free or deeply discounted plans.

**Revenue leakage** — The total value given away through discounts and free trials across all transactions. Tracked monthly in the dashboard to show whether promotional spending is growing relative to actual revenue.

**Skip rate** — The proportion of songs a user started but stopped before reaching 25% of the track. A high skip rate means the user is not finding music they want to listen to. Importantly, this metric tends to rise before a user actually cancels — it is a leading signal, not a trailing one.

**Completion rate** — The proportion of songs a user listened to all the way through. Users who regularly complete songs are far more likely to retain.

**Active days** — The number of calendar days in the observation window on which a user had any listening activity at all. Used as the primary engagement metric since the data does not record session length directly.

**Cohort** — A group of subscribers who registered in the same time period. Cohort analysis compares churn rates across registration years to see whether the product has gotten better or worse at retaining new subscribers over time.

**LTV (lifetime value)** — The total revenue a subscriber generates over the full duration of their relationship with the platform. Users on annual plans or with high engagement have substantially higher LTV than month-to-month, low-engagement subscribers.

---

## Assumptions Made

A few decisions had to be made when the data was not clear cut.

Ages below 5 and above 80 were excluded. The dataset has birth years entered by users during registration, and a number of them are clearly wrong — birth years in the 1800s, future dates, and so on. These were filtered out rather than corrected.

A transaction was classified as promotional if the amount paid was less than 70% of the listed plan price. This threshold was based on the distribution of discount levels in the data — there is a clear split between light discounts and heavy promotional pricing.

A song was counted as skipped if it fell into the `num_25` bucket in the user logs — meaning the user listened to less than 25% of the track. This is not a perfect measure of skipping but it strongly correlates with intentional behaviour in the data.

City codes in the original dataset are anonymous numbers. City 1 was labelled "City 1 (Largest)" because its subscriber volume is dramatically higher than every other city — it is almost certainly Taipei or the greater northern Taiwan metro area. All other cities keep their numerical labels.

Where a user had multiple transactions, revenue was summed across all of them, and behavioural flags like auto-renew were averaged. This means a user who had auto-renew on for 8 out of 10 transactions gets an auto-renew rate of 80%, rather than a simple yes/no.

---

## What Was Found

The overall churn rate in the dataset is around 9%. That means roughly 1 in every 11 subscribers left during the observation window.

The biggest single gap between retained and churned users is auto-renew adoption. Retained users have auto-renew switched on 92% of the time. Churned users only 47% of the time. That is a 45-point difference, which is far larger than any other feature in the dataset.

Skip rate is the most useful early warning signal. Users who skip more than 80% of what they start are churning at nearly three times the base rate. And unlike cancellations, skip rate rises before a user leaves — which means there is time to do something about it.

Subscribers who have spent less than $50 across their entire history with KKBox are the highest-churn group. Those who have crossed $600 in lifetime spend almost never leave. The relationship between money spent and loyalty is consistent and steep.

Heavy promo users — those who received discounted or free transactions more than half the time — churn at 20 to 25%, roughly double the overall rate. These users were not being converted into paying subscribers. They cycled through promotions and left when the deals ran out.

City 1 has a churn rate of 6.4% compared to 12–15% in most other cities. That gap is consistent and large enough that whatever is working in City 1 is worth understanding and replicating elsewhere.

Android-registered users churn at 23%, which is nearly four times the rate of web-registered users at 4.5%. The app is not the problem — the acquisition channel and the user intent that comes with it probably is.

---

## Recommendations

**Stop giving the same people free trials twice.** Users who have already been on a promotional plan should be offered a discounted paid option, not another free trial. A 90-day cooldown on repeat promotions per account would significantly reduce the share of users who are just cycling through free plans with no intention of ever paying full price.

**Make auto-renew adoption a priority.** The data is clear that having auto-renew on is one of the strongest predictors of staying. An in-app prompt offering one extra month free in exchange for enabling auto-renew would likely pay for itself many times over in reduced churn.

**Act on skip rate before users cancel.** When a user's weekly skip rate crosses 75%, the recommendation engine is not working for them. A curated playlist push notification at that point — not a discount, but better content — addresses the actual problem.

**Fix the Android acquisition funnel.** The churn rate among Android-registered users is not a product problem, it is an audience problem. Reviewing what advertising is driving those installs, and adding a personalisation step on first launch, would likely both help.

**Understand City 1 and use that knowledge elsewhere.** A 6.4% churn rate in one city versus 12–15% in others is not random. Before spending money on retention campaigns in high-churn cities, it is worth figuring out what is already working in City 1.

**Push monthly subscribers toward annual plans after the third renewal.** Three consecutive monthly renewals is a reasonable signal of genuine intent. That is the right moment to offer an annual plan at a meaningful discount — not earlier, when commitment is unclear.
