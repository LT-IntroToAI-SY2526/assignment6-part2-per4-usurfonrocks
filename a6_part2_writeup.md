# Assignment 6 Part 2 - Writeup

---

## Question 1: Feature Importance

Based on your house price model, rank the four features from most important to least important. Explain how you determined this ranking.

**YOUR ANSWER:**
1. Most Important: Bedrooms
2. Bathrooms
3. Age
4. Least Important: SquareFeet

**Explanation:**

I looked at the coefficients that the model gave me. The bigger the number, the more important it is. Bedrooms had 6648.97 which was the biggest, then bathrooms had 3858.90, age had 950.35, and square feet only had 121.11 so it was the least important.

---

## Question 2: Interpreting Coefficients

Choose TWO features from your model and explain what their coefficients mean in plain English. For example: "Each additional bedroom increases the price by $___"

**Feature 1:**

Bedrooms - Each additional bedroom increases the price by about $6,649.

**Feature 2:**

Age - Each year older the house is, the price goes down by about $950. So older houses are cheaper which makes sense.

---

## Question 3: Model Performance

What was your model's R² score? What does this tell you about how well your model predicts house prices? Is there room for improvement?

**YOUR ANSWER:**

My R² score was 0.9936. This means the model is really good at predicting prices, like 99% accurate. The RMSE was $4,477 so predictions are usually off by less than $5,000 which is pretty good for house prices. I don't think theres much to improve since its already so high.

---

## Question 4: Adding Features

If you could add TWO more features to improve your house price predictions, what would they be and why?

**Feature 1:**

Location

**Why it would help:**

Where the house is matters a lot. A house in a nice neighborhood is gonna cost way more than the same house somewhere else.

**Feature 2:**

Garage

**Why it would help:**

Having a garage makes a house worth more. People like having somewhere to park their car and store stuff.

---

## Question 5: Model Trust

Would you trust this model to predict the price of a house with 6 bedrooms, 4 bathrooms, 3000 sq ft, and 5 years old? Why or why not? (Hint: Think about the range of your training data)

**YOUR ANSWER:**

I probably wouldn't trust it that much. The training data only had houses up to 5 bedrooms, 3 bathrooms, and 2500 square feet. This house has more bedrooms, more bathrooms, and more square feet than any house the model learned from. So the model is just guessing what happens outside what it knows, which might not be right.


