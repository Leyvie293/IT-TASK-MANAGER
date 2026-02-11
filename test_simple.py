import os
import sys

print("Testing basic setup...")
print("=" * 60)

# Check Python
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Check for required directories
print("\nChecking directories:")
dirs = ['controllers', 'models', 'templates', 'static']
for d in dirs:
    exists = os.path.exists(d)
    print(f"  {d}: {'✓' if exists else '✗'}")

# Check for controller files
print("\nChecking controller files:")
controller_files = ['auth_controller.py', 'task_controller.py', 'report_controller.py']
for f in controller_files:
    path = os.path.join('controllers', f)
    exists = os.path.exists(path)
    print(f"  {f}: {'✓' if exists else '✗'}")
    if exists:
        size = os.path.getsize(path)
        print(f"    Size: {size} bytes")

# Check database file
print("\nChecking database setup:")
try:
    import sqlite3
    # Try to create a test database
    test_db = 'test_db.db'
    conn = sqlite3.connect(test_db)
    conn.execute('CREATE TABLE test (id INTEGER)')
    conn.close()
    os.remove(test_db)
    print("  SQLite: ✓ Working")
except Exception as e:
    print(f"  SQLite: ✗ Error: {e}")

print("\n" + "=" * 60)