###########################################################################
#
# @file postgres.py
#
###########################################################################

import psycopg2
from psycopg2.extensions import AsIs

###########################################################################
#
# A wrapper around the psycopg2 python library.
#
#    The Database class is a high-level wrapper around the psycopg2
#    library. It allows users to create a postgresql database connection and
#    write to or fetch data from the selected database. It also has
#    various utility functions such as getLast(), which retrieves
#    only the very last item in the database, toCSV(), which writes
#    entries from a database to a CSV file, and summary(), a function
#    that takes a dataset and returns only the maximum, minimum and
#    average for each column. The Database can be opened either by passing
#    on the name of the sqlite database in the constructor, or optionally
#    after constructing the database without a name first, the open()
#    method can be used. Additionally, the Database can be opened as a
#    context method, using a 'with .. as' statement. The latter takes
#    care of closing the database.
#
###########################################################################


class Database:

    #######################################################################
    #
    # The constructor of the Database class
    #
    #  The constructor can either be passed the name of the database to open
    #  or not, it is optional. The database can also be opened manually with
    #  the open() method or as a context manager.
    #
    #  @param url Optionally, the url of the database to open.
    #
    #  @see open()
    #
    #######################################################################

    def __init__(self, logger, url=None):

        self.conn = None
        self.cursor = None
        self.logger = logger
        self.url =url
        if url:
            self.connect()

    #######################################################################
    #
    # Opens a new database connection.
    #
    #  This function manually opens a new database connection. The database
    #  can also be opened in the constructor or as a context manager.
    #
    #  @param url The url of the database to open.
    #
    #  @see \__init\__()
    #
    #######################################################################

    def connect(self):
        # Access credentials via the passed on url. The url must
        # be parsed with the urlparse library.
        self.conn = psycopg2.connect(self.url)
    #######################################################################
    #
    # Function to fetch/query data from a database.
    #
    #  This is the main function used to query a database for data.
    #
    #  @param table The name of the database's table to query from.
    #
    #  @param columns The string of columns, comma-separated, to fetch.
    #
    #  @param limit Optionally, a limit of items to fetch.
    #
    #######################################################################

    def selectQuery(self, query, params=None):
        try:
            try:
                cursor = self.conn.cursor()
            except:
                self.connect()
                cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            rows = cursor.fetchall()
            self.conn.commit()
        except Exception as e:
            self.logger.error(e)
            self.conn.rollback()
        if 'rows' in locals():
            return rows
        else:
            return False

    def fetchoneQuery(self, query, params=None):
        try:
            try:
                cursor = self.conn.cursor()
            except:
                self.connect()
                cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            rows = cursor.fetchall()
            self.conn.commit()
        except Exception as e:
            self.logger.error(e)
            self.conn.rollback()
        if 'rows' in locals():
            return rows
        else:
            return False

    def insertQuery(self, query, params=None):
        try:
            try:
                cursor = self.conn.cursor()
            except:
                self.connect()
                cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.conn.commit()
        except Exception as e:
            self.logger.error(e)
            self.conn.rollback()
        if 'e' in locals():
            return True
        else:
            return False


    def insertQueryDict(self, query, columns, values):
        try:
            try:
                cursor = self.conn.cursor()
            except:
                self.connect()
                cursor = self.conn.cursor()

            cursor.execute(query, (AsIs(','.join(columns)), tuple(values)))        
            self.conn.commit()
        except Exception as e:
            print(e)
            self.logger.error(e)
            self.conn.rollback()
        if 'e' in locals():
            return True
        else:
            return False


    def updateQuery(self, query, params=None):
        try:
            try:
                cursor = self.conn.cursor()
            except:
                self.connect()
                cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.conn.commit()
        except Exception as e:
            self.logger.error(e)
            self.conn.rollback()
        if 'e' in locals():
            return True
        else:
            return False
    #######################################################################
    #
    # Utilty function to get the last row of data from a database.
    #
    #  @param table The database's table from which to query.
    #
    #  @param columns The columns which to query.
    #
    #######################################################################


    def closeConn(self):
        try:
            try:
                self.conn.cursor()

            except:
                self.connect()

            self.conn.closeConn()

        except Exception as e:
            self.logger.error(e)
            self.conn.rollback()
        if 'e' in locals():
            return True
        else:
            return False

