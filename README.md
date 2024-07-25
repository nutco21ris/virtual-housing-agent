# San Francisco Smart HomeFinder

This project aims to solve the challenge of finding a suitable rental property in San Francisco by leveraging data science and machine learning techniques. The Virtual Housing Agent provides users with personalized recommendations based on their preferences, and displays results on an interactive map.


## Access the Application

The application has been deployed to Streamlit, and you can access it directly using the following link:

[Virtual Housing Agent San Francisco](https://virtual-housing-agent-sf-nutco21ris.streamlit.app)


## Motivation

I embarked on this project due to the difficulty I faced in finding a suitable home in San Francisco. Given the high rental prices and varying neighborhood characteristics, I used my data science skills to create a recommendation system to help streamline the process.



## Features

- **Real-time Rental Data**: Fetches data from Rental Listings API to provide up-to-date rental listings.
- **Location-based Filtering**: Filters listings based on user preferences for location, price, number of bedrooms/bathrooms, etc.
- **Sentiment Analysis**: Analyzes social media comments and reviews to gauge neighborhood safety, convenience, and public transport availability.
- **Interactive Map**: Displays recommended properties on a map for easy visualization.
- **User-friendly Interface**: Provides a smooth user experience with an intuitive interface.



# For Developers

If you are interested in trying out the application on your local machine, please follow the instructions below. Note that these instructions are intended for those who want to experiment with the application locally.



## API Keys

To use this application, you will need API keys for various services. Here’s how to get them:

1. **Google Maps API Key**:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Create a new project or select an existing project.
   - Navigate to **API & Services** > **Credentials**.
   - Click **Create Credentials** and select **API key**.
   - Enable the required APIs. You will need Google Places API, and Geocoding API.

2. **RentCast API Key**:
   - Visit the [RentCast website](https://www.rentcast.io/) and create an account.
   - After logging in, navigate to your API dashboard.
   - Generate a new API key and select one of the available API plans. RentCast offers a free plan that includes up to 50 API requests per month, which is ideal for testing and development purposes.
   - Add your API key to the `example.env` file as shown below:




## Installation

`git clone https://github.com/nutco21ris/virtual-housing-agent.git`

`cd virtual-housing-agent`

`python -m venv venv`

`source venv/bin/activate`     For Windows, use `venv\Scripts\activate`

`pip install -r requirements.txt`

`streamlit run main.py`


