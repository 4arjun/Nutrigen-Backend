# Vitalia.ai

Vitalia.ai is a powerful mobile application designed to empower users with accurate and timely knowledge about the ingredients in food and cosmetic products. By leveraging AI, machine learning, and advanced scanning technologies, Vitalia.ai ensures users make informed and healthier product choices. With just a snap, the app scrapes the web for detailed ingredient information and delivers insights via a fine-tuned LLM (Large Language Model), tailored to the user's unique health profile.

---

## **Problem Statement**

Consumers face several challenges in understanding the potential health impacts of food and cosmetic products due to:

- Deceptive and unclear labeling practices.
- Lack of access to detailed ingredient information.
- Language barriers and limited health literacy.
- Absence of personalized insights based on individual health conditions.

These challenges lead to uninformed decisions, increased health risks, and a lack of trust in product labeling. Existing solutions often fail to provide the depth of information and personalization required to address these problems effectively.

---

## **Solution**

Vitalia.ai addresses these challenges by offering:

1. **Snap & Scrape**: Simply snap a picture of a product to scrape the web for detailed ingredient and nutritional information.
2. **Barcode & QR Code Scanning**: Instantly retrieve product details using barcode or QR code scanning.
3. **AI-Powered Analysis**: Leverage a fine-tuned LLM to analyze ingredients and provide insights on their potential risks, ill effects, and allergens.
4. **Personalized Recommendations**: Deliver tailored insights based on individual health profiles, such as sugar levels, cholesterol, blood pressure, and more.

---

## **Key Features**

- **Ingredient Analysis via Image or Scan**: Snap a photo of the product or scan its barcode to get a comprehensive breakdown of ingredients and nutritional labels.
- **AI-Driven Insights**: Utilize our fine-tuned LLM to identify hazardous ingredients, long-term health impacts, and recommended usage doses.
- **Allergen Detection**: Automatically detect allergens and warn users based on their health profiles and preferences.
- **Personalized Profiles**: Users can create profiles with details like sugar levels, BP, cholesterol levels, LDL/HDL, fatty acid levels, etc., to receive customized insights.
- **Save & Review**: Save scanned products and their insights for future reference.
- **Community Feedback**: Share reviews and feedback on products to improve collective knowledge.
- **Sustainability Insights**: Highlight eco-friendly and cruelty-free products for sustainable consumer choices.

---

## **How It Works**

1. **Sign-Up/Sign-In**: Users can register or log in to the app. Premium users can provide additional health information for personalized recommendations.
2. **Scan or Snap**: Users can either search for a product name, scan a barcode, or snap a picture of the product.
3. **Data Processing**: The app fetches the entire ingredient and nutritional details from the web and processes it using a fine-tuned LLM.
4. **Insight Generation**: The app provides:
   - Hazardous ingredient classifications.
   - Possible ill effects and long-term impacts.
   - Allergen warnings.
   - Recommended usage doses.
5. **Save & Review**: Users can save the insights for later review or share them with others.

---

## **Tech Stack**

- **Frontend**: React Native
- **Backend**: Django
- **Database**: PostgreSQL
- **AI/ML**: PyTorch, Fine-tuned LLM (Large Language Model)
- **Cloud Services**: AWS
- **Barcode/QR Code Scanning**: OpenCV for barcode detection

---

## **Planned Features**

- **Expanded Multilingual Support**: Provide product insights in more regional languages to enhance accessibility.
- **Integration with Wearables**: Sync health profiles with wearable devices for real-time updates and recommendations.
- **API for Developers**: Offer an API to integrate Vitalia.ai's functionality into other platforms.

---
## **Getting Started**

1. Clone this repository:
   ```bash
   git clone https://github.com/4arjun/Vitalia.ai
   ```

2. Navigate to the project directory:
   ```bash
   cd vitalia-ai
   ```

3. Set up a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Apply migrations to set up the database:
   ```bash
   python manage.py migrate
   ```

6. Run the development server:
   ```bash
   python manage.py runserver
   ```

7. Access the application in your browser at:
   ```
   http://127.0.0.1:8000
   ```

---

## **Resources**

- **YouTube Video Overview**: [Watch Here](https://www.youtube.com/watch?v=FYfEDyfa2xg)
- **Figma Design**: [View on Figma](https://www.figma.com/design/JsK7MAc8TEIT8CKzBjwEdm/Nutrigen?node-id=0-1&p=f&t=uxnOt5yfGxfSVlfe-0)

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
