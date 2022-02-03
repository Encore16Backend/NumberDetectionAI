import sqlite3

conn = sqlite3.connect('dashboard.sqlite')
c = conn.cursor()

c.execute('DROP TABLE IF EXISTS dashboard_db')
c.execute('CREATE TABLE dashboard_db (nickname TEXT, score INTEGER)')

conn.commit()
conn.close()