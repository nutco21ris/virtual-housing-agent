import os
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import webbrowser

# Visualization function
def create_map_visualization(results):
    map_file = "recommended_listings_map.html"
    
    try:
        print(f"Attempting to create map with {len(results)} results")
        
        if isinstance(results, list):
            df = pd.DataFrame(results)
        elif isinstance(results, pd.DataFrame):
            df = results
        else:
            raise ValueError("Invalid input type. Expected DataFrame or list of dictionaries.")
        
        print(f"DataFrame created with {len(df)} rows")
        print(f"Columns in DataFrame: {df.columns}")

        if df.empty:
            print("No listings to display on the map.")
            sf_map = folium.Map(location=[37.7749, -122.4194], zoom_start=12)
        else:
            sf_map = folium.Map(location=[37.7749, -122.4194], zoom_start=12)
            marker_cluster = MarkerCluster().add_to(sf_map)

            for rank, (index, row) in enumerate(df.iterrows(), start=1):
                try:
                    price = row.get('original_price', row['price'])
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=(
                            f"Rank: {rank}<br>"
                            f"Address: {row['formattedAddress']}<br>"
                            f"Price: ${price:.2f}<br>"
                            f"Bedrooms: {row['bedrooms']}<br>"
                            f"Bathrooms: {row['bathrooms']}<br>"
                            f"Score: {row['enhanced_score']:.2f}"
                        ),
                        icon=folium.Icon(color='blue')
                    ).add_to(marker_cluster)
                except Exception as e:
                    print(f"Error adding marker for listing {rank}: {e}")
                    print(f"Row data: {row}")

        sf_map.save(map_file)
        print(f"Map saved to {map_file}")

    except Exception as e:
        print(f"An error occurred while creating the map: {e}")
        import traceback
        traceback.print_exc()
        # Create a simple HTML file with an error message
        with open(map_file, 'w') as f:
            f.write(f"<html><body><h1>Error creating map</h1><p>{str(e)}</p></body></html>")
    
    return os.path.realpath(map_file)

def open_map(map_file):
    if map_file and os.path.exists(map_file):
        webbrowser.open('file://' + os.path.realpath(map_file))
    else:
        print(f"Unable to open map. File not found: {map_file}")