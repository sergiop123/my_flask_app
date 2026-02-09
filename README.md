# Superstore Data Analysis Web Application

A Flask web application for interactive analysis and visualization of Superstore sales data. This application allows users to filter data and run various analytical queries to gain insights into sales performance, profitability, and product discounts.

##  Project Structure

superstore_analysis/
├── app.py # Main Flask application
├── superstore.csv # Dataset file
├── README.md # Project documentation (this file)
├── reflection.md # AI tools usage reflection
├── templates/
│ └── index.html # Web interface template
└── static/
└── style.css # CSS styling file


##  Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation & Setup

1. **Install required dependencies:**
   ```bash
   pip install flask pandas plotly
2. Run the application:
 python app.py
3. Access the application:
Open your web browser and navigate to:
 http://127.0.0.1:5000
 
 ### Features
#Data Filtering
Category: Filter by product categories (Furniture, Office Supplies, Technology)

Sub-Category: Filter by specific product sub-categories

Region: Filter by geographic regions (West, East, Central, South)

Segment: Filter by customer segments (Consumer, Corporate, Home Office)

#Analytical Queries

Dashboard: Shows an overview of many metrics at once using graphs for the selected date range

Total Sales and Profit: Calculate sum of Sales and Profit for filtered data w/ bar graph

Average Discount by Product: Show average discount percentage for each product w/ bar graph

Total Sales by Year: Display yearly sales trends w/ line graph

Profit by Region: Compare profitability across different regions w/bar graph

Products with Negative Profit: Identify loss-making products w/ bar graph

Top Customers by Profit: Displays customers with the highest spending w/ bar graph

Sales by Category: Displays sales by each category w/ bar graph

Monthly Sales Trend: Displays monthly sales numbers and trend w/ line graph

Top Products by Sales: Shows top 15 selling products using sales w/ bar graph

Top Products by Profit Margin: Shows top 15 products using profit margin w/ bar graph

Discount Impact Analysis: Displays how discount affects profit margin w/ a bar graph 

Geo Map: Sales by State: Displays a map by state with states with more sales having a progressively darker fill
 ### How to Use
Apply Filters: Use the dropdown menus to select your desired filters (Category, Sub-Category, Region, Segment)

Choose Query: Select an analytical query from the query dropdown menu

Choose Date Range: Select the data range for the query to run

Run Analysis: Click the "Run Analysis" button

View Results: See formatted results displayed in a table and graph below the form

#️## Technologies Used
Backend: Python, Flask

Data Processing: Pandas

Frontend: HTML, CSS, Plotly

Data Visualization: Table-based results formatting

 ### Dataset Information
The application uses the Superstore dataset which contains:

Sales transaction data

Product information

Customer segments

Geographic regions

Profit and discount information

Order dates and shipping details

  ### Customization
To modify or add new queries:

Edit the app.py file in the main query processing section

Add the chart function in the chart builder section of app.py

Add new query functions following the existing pattern

Update the HTML template in templates/index.html to include new query options

Modify the CSS in static/style.css for any styling changes

  ### Troubleshooting
Common Issues:
"File Not Found" Error:

Ensure superstore.csv is in the main project directory

Check the file name matches exactly (case-sensitive)

Import Errors:

Verify all dependencies are installed: pip install flask pandas plotly

Check Python version compatibility with python --version

Port Already in Use:

The application uses port 5000 by default

Change port by modifying: app.run(debug=True, port=5001)

Template Not Found:

Ensure templates folder exists and contains index.html

Verify static folder exists with style.css

  ### License
This project is for educational purposes as part of a coursework assignment.

  ### Authors:
      Jordan Dixon, Lola Bui, Sergio Parreno

#Course: Bus 4 110-A

#Institution: San Jose State University

### Related Resources
Flask Documentation

Pandas Documentation

Plotly Documentation

Superstore Dataset Information

Note: This project was developed as part of an academic assignment focusing on web application development with Flask and data analysis with Pandas.
