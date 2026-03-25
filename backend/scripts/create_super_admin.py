"""
Script to create a super admin user.
Run this after migration to create the first super admin.

Usage:
    python -m backend.scripts.create_super_admin
"""
import asyncio
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.user import User
from app.models.organization import Organization
from app.core.security import get_password_hash
from uuid import uuid4
import getpass


async def create_super_admin():
    async with AsyncSessionLocal() as db:
        print("=== Create Super Admin User ===\n")
        
        # Get default organization
        org_result = await db.execute(
            select(Organization).where(Organization.name == "default")
        )
        default_org = org_result.scalar_one_or_none()
        
        if not default_org:
            print("ERROR: Default organization not found!")
            print("Please run migrations first: alembic upgrade head")
            return
        
        # Get user input
        email = input("Email: ").strip()
        username = input("Username: ").strip()
        password = getpass.getpass("Password: ")
        password_confirm = getpass.getpass("Confirm Password: ")
        
        if password != password_confirm:
            print("ERROR: Passwords do not match!")
            return
        
        if len(password) < 8:
            print("ERROR: Password must be at least 8 characters!")
            return
        
        # Check if email exists
        result = await db.execute(
            select(User).where(User.email == email)
        )
        if result.scalar_one_or_none():
            print(f"ERROR: User with email {email} already exists!")
            return
        
        # Check if username exists
        result = await db.execute(
            select(User).where(User.username == username)
        )
        if result.scalar_one_or_none():
            print(f"ERROR: Username {username} already taken!")
            return
        
        # Create super admin
        super_admin = User(
            id=uuid4(),
            email=email.lower(),
            username=username,
            hashed_password=get_password_hash(password),
            full_name=input("Full Name (optional): ").strip() or None,
            role="super_admin",
            is_active=True,
            is_verified=True,
            organization_id=default_org.id,
            preferences={"theme": "light", "language": "en"},
            user_metadata={"created_via": "script", "is_initial_super_admin": True}
        )
        
        db.add(super_admin)
        await db.commit()
        await db.refresh(super_admin)
        
        print(f"\n✅ Super admin created successfully!")
        print(f"   Email: {super_admin.email}")
        print(f"   Username: {super_admin.username}")
        print(f"   Role: {super_admin.role}")
        print(f"   Organization: {default_org.display_name}")
        print(f"\nYou can now login with these credentials.")


if __name__ == "__main__":
    asyncio.run(create_super_admin())