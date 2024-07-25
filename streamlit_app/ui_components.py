import streamlit as st
import folium
from streamlit_folium import folium_static

def format_price(price):
    if price and price >= 100:
        return f"${price:,.2f}"
    else:
        return "Price not available"
    

def render_chat_interface():
    chat_container = st.container()

    with chat_container:
        for role, message in st.session_state.messages:
            if role == "assistant":
                st.markdown(f'<div class="chat-message assistant-message">{message}</div>', unsafe_allow_html=True)
            elif role == "user" and not message.startswith('{"budget":'):
                st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)



def create_map_with_recommendations(recommendations):
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12)
    
    for i, rec in enumerate(recommendations, 1):
        if 'latitude' in rec and 'longitude' in rec:
            price_display = format_price(rec.get('price'))
            
            popup_content = f"""
            <b>Rank: {i}</b><br>
            Address: {rec.get('formattedAddress', 'N/A')}<br>
            Price: {price_display}<br>
            Bedrooms: {rec.get('bedrooms', 'N/A')}<br>
            Bathrooms: {rec.get('bathrooms', 'N/A')}<br>
            Score: {rec.get('enhanced_score', 'N/A'):.2f}
            """
            
            folium.Marker(
                [rec['latitude'], rec['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Rank {i}: {price_display}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
    
    folium_static(m)


def render_recommendations(recommendations):
    if not recommendations:
        st.write("No recommendations found based on the given criteria.")
    else:
        user_info = st.session_state.user_info
        st.write(f"Showing results for: {user_info['bedrooms']} bedroom(s), "
                 f"${user_info['budget'][0]}-{user_info['budget'][1]}, "
                 f"near {user_info['location']}")
        
        st.subheader("Recommended Listings")
        
        view_type = st.session_state.user_info.get('view_type', 'Map')
        
        if view_type == 'Map':
            create_map_with_recommendations(recommendations)
            with st.expander("View Detailed List"):
                display_detailed_list(recommendations)
        else:
            display_detailed_list(recommendations)


def display_detailed_list(recommendations):
    for i, rec in enumerate(recommendations, 1):
        st.write(f"**Rank {i}**")
        st.write(f"Address: {rec.get('formattedAddress', 'N/A')}")
        st.write(f"Price: {format_price(rec.get('price'))}")
        
        price = rec.get('price')
        if price and price > 0:
            st.write(f"Price: ${price:,.2f}")
        else:
            st.write("Price: Not available")
        
        st.write(f"Bedrooms: {rec.get('bedrooms', 'N/A')}")
        st.write(f"Bathrooms: {rec.get('bathrooms', 'N/A')}")
        st.write(f"Score: {rec.get('enhanced_score', 'N/A'):.2f}")
        st.write("---")