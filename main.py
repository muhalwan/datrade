import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

# Define the path to your JSON file
json_file_path = 'crypto_trading.order_book.json'

# Check if the file exists
if not os.path.exists(json_file_path):
    raise FileNotFoundError(f"The file {json_file_path} does not exist. Please check the path and try again.")

# Function to parse the JSON data
def parse_json(file_path):
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}")

    parsed_data = []
    for entry in data:
        try:
            parsed_entry = {
                "id": entry["_id"]["$oid"],
                "timestamp": entry["timestamp"]["$date"],
                "symbol": entry["symbol"],
                "side": entry["side"],
                "price": float(entry["price"]),
                "quantity": float(entry["quantity"]),
                "update_id": int(entry["update_id"])
            }
            parsed_data.append(parsed_entry)
        except KeyError as e:
            print(f"Missing key {e} in entry {entry}")
        except (TypeError, ValueError) as e:
            print(f"Type error {e} in entry {entry}")
    return parsed_data

# Parse the JSON data
data = parse_json(json_file_path)

# Check if data is not empty
if not data:
    raise ValueError("No valid data found in the JSON file.")

# Convert to Pandas DataFrame
df = pd.DataFrame(data)

# Convert timestamp strings to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort the DataFrame by timestamp
df.sort_values('timestamp', inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Create the Plotly figure
fig = go.Figure()

# Add Price trace
fig.add_trace(
    go.Scatter(
        x=df['timestamp'],
        y=df['price'],
        mode='lines+markers',
        name='Price',
        line=dict(color='blue'),
        marker=dict(size=8)
    )
)

# Add Quantity trace with a secondary y-axis
fig.add_trace(
    go.Bar(
        x=df['timestamp'],
        y=df['quantity'],
        name='Quantity',
        marker=dict(color='orange'),
        opacity=0.6,
        yaxis='y2'
    )
)

# Update layout for secondary y-axis
fig.update_layout(
    title='BTCUSDT Bid Orders Over Time',
    xaxis_title='Timestamp',
    yaxis=dict(
        title='Price (USDT)',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue')
    ),
    yaxis2=dict(
        title='Quantity',
        titlefont=dict(color='orange'),
        tickfont=dict(color='orange'),
        overlaying='y',
        side='right'
    ),
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor='rgba(255,255,255,0)',
        bordercolor='rgba(255,255,255,0)'
    ),
    template='plotly_dark',
    hovermode='x unified',
    margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins if necessary
)

# Optional: Update x-axis for better date formatting
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1h", step="hour", stepmode="backward"),
            dict(count=6, label="6h", step="hour", stepmode="backward"),
            dict(count=12, label="12h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    ),
    tickformat="%Y-%m-%d %H:%M:%S",
    tickangle=45
)

# Show the figure
fig.show()
