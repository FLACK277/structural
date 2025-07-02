import sqlite3
conn = sqlite3.connect('learning_system.db')
cursor = conn.cursor()
for row in cursor.execute('SELECT user_id, profile_data FROM user_profiles'):
    print(row)
conn.close()