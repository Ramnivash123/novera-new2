"""
Script to initialize Alembic and create the first migration.
Run this once to set up database versioning.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> None:
    """Execute a shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"⚠️  Warnings: {result.stderr}")
        print(f"✅ {description} completed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


def main():
    """Initialize Alembic migrations."""
    
    # Ensure we're in the backend directory
    backend_dir = Path(__file__).parent.parent
    print(f"Working directory: {backend_dir}")
    
    # Step 1: Initialize Alembic (if not already done)
    alembic_dir = backend_dir / "alembic"
    if not alembic_dir.exists():
        run_command(
            ["alembic", "init", "alembic"],
            "Initializing Alembic"
        )
    else:
        print("⏭️  Alembic already initialized, skipping...\n")
    
    # Step 2: Create initial migration
    run_command(
        ["alembic", "revision", "--autogenerate", "-m", "Initial schema with pgvector"],
        "Creating initial migration"
    )
    
    print("\n" + "="*60)
    print("🎉 Migration setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated migration in alembic/versions/")
    print("2. Apply migration: alembic upgrade head")
    print("3. Start building your application!\n")


if __name__ == "__main__":
    main()