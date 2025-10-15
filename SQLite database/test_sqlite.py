import sqlite3
from faker import Faker
import random

# Initialize Faker for generating fake data
fake = Faker()

# Connect to SQLite (creates file if it doesn't exist)
conn = sqlite3.connect("students.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    age INTEGER,
    email TEXT UNIQUE,
    major TEXT,
    gpa REAL
)
""")

print("✅ Table 'students' created successfully.")

# Generate and insert fake data
majors = ["Computer Science", "Mathematics", "Economics", "Physics", "History", "Psychology"]

students = [
    (fake.name(),
     random.randint(18, 25),
     fake.email(),
     random.choice(majors),
     round(random.uniform(2.0, 4.0), 2))
    for _ in range(100)
]

cursor.executemany("""
INSERT INTO students (full_name, age, email, major, gpa)
VALUES (?, ?, ?, ?, ?)
""", students)

conn.commit()
print(f"✅ Inserted {len(students)} fake student records.")

# Fetch and display sample data
cursor.execute("SELECT * FROM students LIMIT 5")
rows = cursor.fetchall()

print("\n🎓 Sample Data:")
for row in rows:
    print(row)

# Close connection
conn.close()
print("\n✅ Connection closed successfully.")
