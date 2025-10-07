# Quick Start: HuggingFace Spaces Deployment

This guide will help you deploy the Bank Marketing model to HuggingFace Spaces in under 10 minutes.

## Prerequisites

- HuggingFace account ([sign up](https://huggingface.co/join))
- Git installed
- Git LFS installed ([download](https://git-lfs.github.com/))

## Step 1: Create a Space

1. Go to https://huggingface.co/new-space
2. Fill in details:
   - **Owner**: Your username or organization
   - **Space name**: `bank-marketing-prediction`
   - **License**: MIT
   - **Space SDK**: Gradio
   - **Space hardware**: CPU basic (free)
3. Click "Create Space"

## Step 2: Clone the Space Repository

```bash
# Clone your newly created space
git clone https://huggingface.co/spaces/<your-username>/bank-marketing-prediction
cd bank-marketing-prediction
```

## Step 3: Copy Files

```bash
# From this repository root, copy files to the Space directory
# Adjust paths based on your location

# Copy application files
cp /path/to/this/repo/huggingface_space/app.py .
cp /path/to/this/repo/huggingface_space/requirements.txt .
cp /path/to/this/repo/huggingface_space/README.md .
cp /path/to/this/repo/huggingface_space/.gitattributes .

# Copy model files
cp /path/to/this/repo/models/lightgbm_retrained_tuned.pkl .

# Copy preprocessing files
mkdir -p preprocessing
cp /path/to/this/repo/models/preprocessing/scaler.pkl preprocessing/
cp /path/to/this/repo/models/preprocessing/label_encoders.pkl preprocessing/
```

## Step 4: Initialize Git LFS

```bash
# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"

# Add .gitattributes
git add .gitattributes
```

## Step 5: Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment of bank marketing prediction model"

# Push to HuggingFace
git push
```

## Step 6: Wait for Build

- Go to your Space URL: `https://huggingface.co/spaces/<your-username>/bank-marketing-prediction`
- Wait for the build to complete (1-2 minutes)
- The Space will automatically start when ready

## Step 7: Test Your Space

1. Open the Space URL in your browser
2. Fill in the form with sample data:
   - Age: 30
   - Job: admin.
   - Marital: single
   - Education: university.degree
   - (Leave others as default)
3. Click "Predict Subscription"
4. You should see the prediction result!

## Automated Deployment (Optional)

Set up GitHub Actions for automatic deployment:

### 1. Create HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "GitHub Actions Deploy"
4. Type: Write
5. Copy the token

### 2. Add to GitHub Secrets

1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `HF_TOKEN`
5. Value: [paste your token]
6. Click "Add secret"

### 3. Update Workflow

Edit `.github/workflows/deploy-huggingface.yml`:

```yaml
# Replace <your-username> with your HuggingFace username
repo_id='<your-username>/bank-marketing-prediction'
```

### 4. Trigger Deployment

```bash
# Make any change to huggingface_space/ and push
git add .
git commit -m "Update Space"
git push
```

The GitHub Action will automatically deploy to HuggingFace Spaces!

## Troubleshooting

### Issue: "Git LFS file too large"

**Solution**: Free tier has 10GB limit. Our model files are under 2GB, so this shouldn't occur. If it does:
- Upgrade to Pro tier
- Or use smaller model files

### Issue: "Module not found"

**Solution**: Check `requirements.txt` has all dependencies:
```
gradio==4.7.1
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
lightgbm==4.1.0
joblib==1.3.2
```

### Issue: "Model file not found"

**Solution**: Verify file structure:
```
bank-marketing-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitattributes
├── lightgbm_retrained_tuned.pkl
└── preprocessing/
    ├── scaler.pkl
    └── label_encoders.pkl
```

### Issue: "Space build fails"

**Solution**: Check build logs in the Space UI. Common issues:
- Missing dependencies in requirements.txt
- Incorrect file paths in app.py
- Python version mismatch (use 3.10)

## Next Steps

1. ✅ Share your Space with others
2. ✅ Embed in websites using iframe
3. ✅ Set up monitoring
4. ✅ Add authentication (upgrade to Pro)
5. ✅ Connect to API for programmatic access

## Resources

- [HuggingFace Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [Full Deployment Guide](DEPLOYMENT_GUIDE.md)

## Support

Having issues? 
- Check the [Deployment Guide](DEPLOYMENT_GUIDE.md)
- Open an issue on GitHub
- Ask on HuggingFace forums

Happy deploying! 🚀
