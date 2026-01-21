"""Extract SWEET selection2 user data."""

from pathlib import Path
import zipfile

selection2_zip_dir = Path("data/SWEET/selection2/selection2_zip")
users_output_dir = Path("data/SWEET/selection2/users")

users_output_dir.mkdir(parents=True, exist_ok=True)

user_zips = sorted(selection2_zip_dir.glob("user*.zip"))
print(f"Found {len(user_zips)} user zip files in selection2")

for user_zip in user_zips:
    user_name = user_zip.stem  # e.g., "user0085"
    user_dir = users_output_dir / user_name

    if user_dir.exists() and list(user_dir.glob("*.csv")):
        print(f"✓ {user_name} already extracted")
        continue

    print(f"Extracting {user_name}...", end=" ")
    with zipfile.ZipFile(user_zip, "r") as zip_ref:
        zip_ref.extractall(user_dir)
    print("done")

print(f"\n✅ All {len(user_zips)} users extracted to {users_output_dir}")
