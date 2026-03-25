"""
Script to promote an existing admin user to super admin.

Usage:
    python -m backend.scripts.promote_to_super_admin
"""
import asyncio
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.user import User


async def promote_to_super_admin():
    async with AsyncSessionLocal() as db:
        print("=== Promote User to Super Admin ===\n")
        
        # Get all admins
        result = await db.execute(
            select(User).where(User.role.in_(["admin", "org_admin"]))
        )
        admins = result.scalars().all()
        
        if not admins:
            print("No admin users found!")
            return
        
        print("Existing admin users:")
        for i, admin in enumerate(admins, 1):
            print(f"{i}. {admin.email} ({admin.username}) - Role: {admin.role}")
        
        choice = input("\nSelect user number to promote (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return
        
        try:
            index = int(choice) - 1
            if index < 0 or index >= len(admins):
                print("Invalid selection!")
                return
        except ValueError:
            print("Invalid input!")
            return
        
        selected_user = admins[index]
        
        confirm = input(f"\nPromote {selected_user.email} to super admin? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print("Cancelled.")
            return
        
        # Promote to super admin
        selected_user.role = "super_admin"
        await db.commit()
        
        print(f"\n✅ {selected_user.email} promoted to super admin!")


if __name__ == "__main__":
    asyncio.run(promote_to_super_admin())