# GARUDA GitHub Repository Setup Guide

## Quick Setup (Choose One Method)

### Method 1: Using Setup Script (Recommended)

```bash
cd ~/garuda
chmod +x setup_github_repo.sh
./setup_github_repo.sh
```

This interactive script will guide you through the process.

---

### Method 2: Manual Steps

#### Step 1: Create Repository on GitHub

1. Go to **https://github.com/new**
2. Fill in:
   - **Repository name**: `garuda`
   - **Description**: `GARUDA: Geothermal And Reservoir Understanding with Data-driven Analytics`
   - **Visibility**: Public (recommended for open-source)
   - **⚠️ DO NOT** check "Add README", "Add .gitignore", or "Choose license"
3. Click **"Create repository"**

#### Step 2: Push from Terminal

```bash
cd ~/garuda

# Add GitHub remote
git remote add origin https://github.com/zakusworo/garuda.git

# Push to GitHub
git push -u origin main
```

#### Step 3: Verify

Visit: **https://github.com/zakusworo/garuda**

---

### Method 3: Using GitHub CLI (If Installed)

```bash
cd ~/garuda

# Create repository
gh repo create garuda --public --source=. --remote=origin --push

# Or if already created
gh repo create garuda --public --source=. --remote=origin
git push -u origin main
```

---

## Authentication Options

### Option A: HTTPS with Personal Access Token

1. Generate token: https://github.com/settings/tokens
2. Scopes needed: `repo` (full control of private repositories)
3. When pushing, use token as password:
   ```
   Username: zakusworo
   Password: [your_token]
   ```

### Option B: SSH (Recommended)

1. Generate SSH key (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "greataji13@gmail.com"
   ```

2. Add to GitHub:
   - Copy key: `cat ~/.ssh/id_ed25519.pub`
   - Go to: https://github.com/settings/keys
   - Click "New SSH key"

3. Use SSH remote:
   ```bash
   git remote set-url origin git@github.com:zakusworo/garuda.git
   git push -u origin main
   ```

### Option C: GitHub CLI

```bash
gh auth login
# Follow prompts
gh repo create garuda --public --source=. --remote=origin --push
```

---

## After Pushing

### 1. Add Repository Topics

Go to repository homepage → Click "⚙️" (settings icon) next to "About" → Add topics:

```
reservoir-simulation
geothermal
petroleum
python
computational-physics
indonesia
renewable-energy
finite-volume
iapws-97
```

### 2. Enable GitHub Actions

1. Go to: https://github.com/zakusworo/garuda/actions
2. Click "I understand my workflows, go ahead and enable them"
3. CI will run automatically on future pushes

### 3. Protect Main Branch

1. Settings → Branches → Add branch protection rule
2. Branch name pattern: `main`
3. Check "Require a pull request before merging"
4. Check "Require status checks to pass before merging"

### 4. Add Badges to README

After enabling Actions, add to README.md:

```markdown
[![CI](https://github.com/zakusworo/garuda/actions/workflows/ci.yml/badge.svg)](https://github.com/zakusworo/garuda/actions)
```

---

## Troubleshooting

### Error: "remote: Repository not found"

**Solution**: Repository doesn't exist yet. Create it at https://github.com/new

### Error: "Authentication failed"

**Solutions**:
1. Use personal access token instead of password
2. Or switch to SSH: `git remote set-url origin git@github.com:zakusworo/garuda.git`
3. Or use GitHub CLI: `gh auth login`

### Error: "Updates were rejected because the remote contains work that you do not have"

**Solution**: Force push (only if remote is empty):
```bash
git push -u origin main --force
```

### Error: "Permission denied (publickey)"

**Solution**: SSH key not configured
```bash
# Test SSH connection
ssh -T git@github.com

# If fails, add SSH key to GitHub
cat ~/.ssh/id_ed25519.pub
# Copy output and add to: https://github.com/settings/keys
```

---

## Repository URL

After setup, your repository will be at:

**https://github.com/zakusworo/garuda**

---

## Next Steps After Pushing

1. ✅ Share repository link with collaborators
2. ✅ Add Zenodo DOI (for academic citation)
3. ✅ Set up ReadTheDocs documentation
4. ✅ Register package on PyPI (`garuda-sim`)
5. ✅ Announce on social media / research networks

---

## Citation

Add to repository description or README:

> **Cite as**: Kusworo, Z.A. (2026). GARUDA: Geothermal And Reservoir Understanding with Data-driven Analytics [Computer software]. GitHub. https://github.com/zakusworo/garuda

---

**Good luck! 🦅🇮🇩**
