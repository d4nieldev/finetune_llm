# Database Creation

Before evaluating the QPL pipeline, you need to first set up a SQL database and download the spider sql data.

## Database selection

The recommended approach is to set an Azure SQL Database (free).

Once you have a working SQL server, create a database in that server, and **replace the connection string in the `create_db.py` script** with your connection string.

## Spider

To download the spider sql data, visit the [spider website](https://yale-lily.github.io/spider) and download it from there. Then extract it to your preferred location.

## Create The Database

To create the database, run `python create_db.py -s path_to_extracted_spider_data`. This will take some time.
