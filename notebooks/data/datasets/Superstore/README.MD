
### Superstore Dataset

This is a superstore dataset I created using the components of it
latest version: /Superstore/data_prepared_with_tableau_prep/data_prep/orders_oct27_assigned_unique_productIDs.csv

#### Data Prep
Initial data was prepared in Tableau Prep Builder

Cleaning steps for separate data sources:
- Orders Central
    - Created the “Region” calculated field
    - Created the “Order Date” and “Ship Date” calculated fields and removed partial date
    fields(order year, order month, order day, …)
    - Replaced None values with 0 in “Discounts” field, and changed its data type to Number
    (decimal)
- Orders West
    - Removed the duplicate fields with the prefix “Right_” (20 fields in total)
    - Grouped values of states with their abbreviations.
- Orders East
    - Removed letters “USD” from Sales numbers, and changed its data type to Number (decimal)
- Orders South
    - None, they were following the same format and unioned directly.
- returns_new
    - Removed fields that are not uniquely related to the returns (“Row ID”, “Order Date”,
    “Sub-Category”, “Manufacturer”, “Product Name”)
    - Cleaned “Notes” field. Trimmed spaces and splitted into two fields “Return Notes” and “Return
    Approver”. Deleted the old “Notes” field, grouped Approver names’ different phrasings.
- Quota
    - Added the prefix “Quota-” to each field(“2015”, “2016”, “2017”, “2018”) to present more
    precisely.
- Orders(unioned) + Returns (Left Join)
    - Removed duplicated joined keys “Order ID-1” and “Product ID-1”
    - Added features “Returned?” and “Days to Ship”
- Orders (unioned) + Returns + Quota (Left Join)
    - Removed duplicate records (16302 to 9952)
    - Removed the extra field “Region-1”


After analyzing this data, a conflict with product IDs linking to different product names was investigated and a new version was created by assigning unique product IDs to each product ID - product Name pair based on this notebook:
/Superstore/data_prepared_with_tableau_prep/data_prep/Version_Oct_27_Product_ID_Conflict_Investigation_and_New_ID_Assignments_(Superstore_Dataset).ipynb
