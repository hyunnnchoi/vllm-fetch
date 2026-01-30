#!/bin/bash

# NOTE, hyunnnchoi, 2026.01.30
# Replace lmcache 0.3.12 with latest dev version from GitHub
# Usage: ./fetch.sh [fetch|rollback]

set -e  # Exit on error

# Variables
OLD_LMCACHE="/lmcache"
BACKUP_LMCACHE="/lmcache_bak"
NEW_LMCACHE="/home/work/hyunmokchoi/lmcache-fetch/lmcache"

# Function to fetch new lmcache
fetch_lmcache() {
    echo "=== Starting lmcache fetch mode ==="

    # Check if new lmcache exists
    if [ ! -d "$NEW_LMCACHE" ]; then
        echo "Error: New lmcache directory not found at $NEW_LMCACHE"
        exit 1
    fi

    # Check if old lmcache exists
    if [ ! -d "$OLD_LMCACHE" ]; then
        echo "Error: Old lmcache directory not found at $OLD_LMCACHE"
        exit 1
    fi

    # Remove old backup if exists
    if [ -d "$BACKUP_LMCACHE" ]; then
        echo "Removing existing backup at $BACKUP_LMCACHE..."
        rm -rf "$BACKUP_LMCACHE"
    fi

    # Backup old lmcache
    echo "Backing up $OLD_LMCACHE to $BACKUP_LMCACHE..."
    mv "$OLD_LMCACHE" "$BACKUP_LMCACHE"

    # Move new lmcache to /lmcache
    echo "Moving new lmcache from $NEW_LMCACHE to $OLD_LMCACHE..."
    mv "$NEW_LMCACHE" "$OLD_LMCACHE"

    echo "✓ Done! lmcache has been successfully replaced."
    echo "  Old version backed up at: $BACKUP_LMCACHE"
    echo "  New version installed at: $OLD_LMCACHE"
}

# Function to rollback to backup
rollback_lmcache() {
    echo "=== Starting lmcache rollback mode ==="

    # Check if backup exists
    if [ ! -d "$BACKUP_LMCACHE" ]; then
        echo "Error: Backup directory not found at $BACKUP_LMCACHE"
        echo "Nothing to rollback."
        exit 1
    fi

    # Check if current lmcache exists
    if [ ! -d "$OLD_LMCACHE" ]; then
        echo "Warning: Current lmcache directory not found at $OLD_LMCACHE"
    else
        # Remove current lmcache
        echo "Removing current lmcache at $OLD_LMCACHE..."
        rm -rf "$OLD_LMCACHE"
    fi

    # Restore backup
    echo "Restoring backup from $BACKUP_LMCACHE to $OLD_LMCACHE..."
    mv "$BACKUP_LMCACHE" "$OLD_LMCACHE"

    echo "✓ Done! lmcache has been successfully rolled back."
    echo "  Restored version at: $OLD_LMCACHE"
}

# Main logic
MODE="${1:-fetch}"

case "$MODE" in
    fetch)
        fetch_lmcache
        ;;
    rollback)
        rollback_lmcache
        ;;
    *)
        echo "Usage: $0 [fetch|rollback]"
        echo "  fetch    - Replace old lmcache with new version (default)"
        echo "  rollback - Restore backup version"
        exit 1
        ;;
esac
