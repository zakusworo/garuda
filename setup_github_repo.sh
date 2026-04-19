#!/bin/bash
# GARUDA Repository Setup Script
# Run this to create GitHub repo and push code

set -e  # Exit on error

echo "============================================================"
echo "GARUDA Repository Setup"
echo "============================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REPO_NAME="garuda"
REPO_OWNER="zakusworo"
REMOTE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}.git"

echo -e "${YELLOW}Step 1: Check Git Configuration${NC}"
echo "------------------------------------------------------------"

# Check if git is configured
if ! git config user.name > /dev/null 2>&1; then
    echo -e "${RED}❌ Git user.name not configured${NC}"
    echo "Run: git config --global user.name \"Your Name\""
    exit 1
else
    echo -e "${GREEN}✅ Git user.name configured${NC}"
fi

if ! git config user.email > /dev/null 2>&1; then
    echo -e "${RED}❌ Git user.email not configured${NC}"
    echo "Run: git config --global user.email \"your@email.com\""
    exit 1
else
    echo -e "${GREEN}✅ Git user.email configured${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Verify Local Repository${NC}"
echo "------------------------------------------------------------"

if [ ! -d ".git" ]; then
    echo -e "${RED}❌ Not a git repository${NC}"
    echo "Run: git init"
    exit 1
else
    echo -e "${GREEN}✅ Git repository exists${NC}"
fi

# Check if there are uncommitted changes
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Uncommitted changes detected${NC}"
    read -p "Commit changes before pushing? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "WIP: Uncommitted changes"
        echo -e "${GREEN}✅ Changes committed${NC}"
    fi
fi

echo ""
echo -e "${YELLOW}Step 3: Create GitHub Repository${NC}"
echo "------------------------------------------------------------"
echo ""
echo "Please create the repository on GitHub:"
echo ""
echo "  1. Go to: https://github.com/new"
echo "  2. Repository name: ${REPO_NAME}"
echo "  3. Description: GARUDA - Geothermal And Reservoir Understanding with Data-driven Analytics"
echo "  4. Visibility: Public (recommended for open-source)"
echo "  5. ⚠️  DO NOT initialize with README, .gitignore, or license"
echo "  6. Click 'Create repository'"
echo ""
read -p "Press Enter after you've created the repository..."

echo ""
echo -e "${YELLOW}Step 4: Add Remote and Push${NC}"
echo "------------------------------------------------------------"

# Check if remote already exists
if git remote | grep -q "^origin$"; then
    echo -e "${YELLOW}⚠️  Remote 'origin' already exists${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote remove origin
        echo "Removed existing remote"
    else
        echo "Keeping existing remote"
    fi
fi

if ! git remote | grep -q "^origin$"; then
    echo "Adding remote: ${REMOTE_URL}"
    git remote add origin ${REMOTE_URL}
    echo -e "${GREEN}✅ Remote added${NC}"
fi

echo ""
echo "Pushing to GitHub..."
echo ""

# Try to push
if git push -u origin main 2>&1; then
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}✅ SUCCESS! Repository pushed to GitHub${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo "Your repository is now live at:"
    echo "  👉 https://github.com/${REPO_OWNER}/${REPO_NAME}"
    echo ""
    echo "Next steps:"
    echo "  1. Add repository topics: reservoir-simulation, geothermal, petroleum, python, indonesia"
    echo "  2. Enable GitHub Actions (for CI/CD)"
    echo "  3. Add Zenodo DOI badge (after archiving)"
    echo "  4. Share with collaborators!"
    echo ""
else
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}❌ PUSH FAILED${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    echo "Possible solutions:"
    echo ""
    echo "1. Authentication error:"
    echo "   - Use GitHub CLI: gh auth login"
    echo "   - Or use SSH: git remote set-url origin git@github.com:${REPO_OWNER}/${REPO_NAME}.git"
    echo ""
    echo "2. Repository doesn't exist:"
    echo "   - Create it at: https://github.com/new"
    echo "   - Name: ${REPO_NAME}"
    echo ""
    echo "3. Permission error:"
    echo "   - Check GitHub account: ${REPO_OWNER}"
    echo "   - Verify repository ownership"
    echo ""
fi

echo ""
echo "============================================================"
echo "Setup script completed"
echo "============================================================"
