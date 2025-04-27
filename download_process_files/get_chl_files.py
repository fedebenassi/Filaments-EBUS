import datetime

# Base URL
base_url = "ftp://oc-cci-data:ELaiWai8ae@ftp.rsg.pml.ac.uk/occci-v6.0/geographic/netcdf/daily/chlor_a/"

# Years to loop through
years = [y for y in range(2003, 2024)]

# Open the file to write the URLs
with open('chl_data_urls.txt', 'w') as file:
    # Loop through the years
    for year in years:
        # Loop through each day of the year
        for month in range(1, 13):  # months from 1 to 12
            for day in range(1, 32):  # days from 1 to 31
                try:
                    # Create a valid date to avoid issues with invalid days/months
                    date = datetime.date(year, month, day)
                    
                    # Format date as YYYYMMDD
                    date_str = date.strftime("%Y%m%d")
                    
                    # Construct the full URL
                    url = f"{base_url}{year}/ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-{date_str}-fv6.0.nc"
                    
                    # Write the URL to the file
                    file.write(url + '\n')

                except ValueError:
                    # Skip invalid days, such as Feb 30 or Apr 31
                    continue

print("URLs have been written to 'chl_data_urls.txt'.")