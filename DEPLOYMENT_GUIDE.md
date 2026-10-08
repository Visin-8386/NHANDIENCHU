# Deployment Guide - Handwritten Text Recognition

## 🚀 Quick Start

### Prerequisites
- Git repository connected to GitHub
- Render.com account linked to GitHub repository
- Model file auto-downloads from GitHub Releases on first run

### Automatic Deployment (Recommended)
Every push to `master` branch automatically triggers Render redeploy:
```bash
git push origin master
```

---

## 📋 Pre-Deployment Checklist

### Code Quality
- [ ] Run tests (if any)
- [ ] Check for console errors
- [ ] Verify responsive design on mobile
- [ ] Test dark/light theme toggle
- [ ] Verify all buttons and controls work

### Performance
- [ ] Static assets cached properly
- [ ] Images are compressed
- [ ] Model lazy loading working
- [ ] No console warnings

### Documentation
- [ ] Commit messages are clear
- [ ] Code comments where necessary
- [ ] OPTIMIZATION_REPORT.md updated

---

## 🔧 Manual Deployment Steps

### Step 1: Prepare Changes
```bash
cd d:\WEB_AI
git status  # Check for uncommitted changes
git add .   # Stage changes
git commit -m "meaningful commit message"
```

### Step 2: Push to GitHub
```bash
git push origin master
```

### Step 3: Monitor Deployment
1. Visit Render Dashboard: https://dashboard.render.com/
2. Select your service (nhandienchu)
3. Watch deployment logs in real-time
4. Wait for "Service is live" message

### Step 4: Verify Live Version
1. Visit https://nhandienchu.onrender.com/
2. Do a hard refresh (Ctrl+Shift+R / Cmd+Shift+R)
3. Test core functionality:
   - Draw on canvas
   - Click "Nhận diện" button
   - Check if result appears

---

## ⏱️ Expected Deployment Times

| Stage | Duration | Notes |
|-------|----------|-------|
| GitHub Push | <1s | Immediate |
| Render Detection | <1min | Automatic webhook |
| Build Start | <1min | Queued build |
| Build Phase | 2-3min | Dependencies install |
| Deploy Phase | 1-2min | Service restart |
| Health Check | 30-60s | Warmup request |
| **Total** | **5-8min** | Typical deployment |

---

## 🔍 Troubleshooting

### Deployment Fails
1. Check Render logs for errors
2. Verify `requirements.txt` is correct
3. Check environment variables are set
4. Ensure model download URL is accessible

### Site Shows Old Version
- Hard refresh browser (Ctrl+Shift+R)
- Clear browser cache
- Wait 30 seconds and retry
- Check browser DevTools → Network → Disable cache

### Slow First Request
- **Expected behavior** on Render free tier
- First request after app sleep takes 1-2 minutes
- Subsequent requests are fast (cached model)
- App has 280s timeout to complete

### API Errors
1. Check network tab in browser DevTools
2. Verify `/predict_handwriting` endpoint responds
3. Test with simple drawing first
4. Check Render logs for Python errors

---

## 🛠️ Local Testing Before Deploy

### Setup Local Environment
```bash
pip install -r requirements.txt
python app.py
```

### Test Locally
1. Open http://localhost:5000
2. Draw something
3. Click "Nhận diện"
4. Verify result appears
5. Check console for errors

### Test CSS Changes
1. Edit `static/style.css`
2. Reload browser (hard refresh)
3. Verify changes appear immediately

---

## 📊 Performance Monitoring

### Browser DevTools (Network Tab)
- **Document**: HTML file size
- **JS**: JavaScript bundle size
- **CSS**: Stylesheet size
- **XHR/Fetch**: API response sizes
- **Cache Status**: Should show "from disk cache" on repeat visits

### Render Metrics
- Visit https://dashboard.render.com/
- Check CPU, Memory, and Bandwidth usage
- Monitor for out-of-memory or CPU throttling

### Key Metrics to Monitor
- **FCP (First Contentful Paint)**: <2s target
- **LCP (Largest Contentful Paint)**: <4s target
- **CLS (Cumulative Layout Shift)**: <0.1 target
- **API Response Time**: <5s typical

---

## 🔐 Security Notes

### Secrets Management
- Environment variables stored in Render dashboard
- Never commit secrets to Git
- Use `.env` for local development only
- Add `.env` to `.gitignore`

### Current Environment Variables (Set in Render)
```
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
PYTHONUNBUFFERED=1
```

### Model Downloads
- Model downloads from GitHub Releases
- Verify file size (~29MB) on first deployment
- File cached after download

---

## 📝 Common Commits

### CSS Changes
```bash
git commit -m "🎨 Update: [description of CSS changes]"
```

### Performance Improvements
```bash
git commit -m "🚀 Optimize: [description of performance change]"
```

### Bug Fixes
```bash
git commit -m "🐛 Fix: [description of bug fix]"
```

### Features
```bash
git commit -m "✨ Feature: [description of new feature]"
```

---

## 🔄 Rollback Procedure

### If Deployment Breaks Production

1. **Find Last Good Commit**
   ```bash
   git log --oneline -10  # Show last 10 commits
   ```

2. **Identify Good Version**
   - Look at commit messages
   - Check timestamps

3. **Revert to Previous Version**
   ```bash
   git revert HEAD -m "Revert to stable version"
   git push origin master
   # or
   git reset --hard <commit-hash>
   git push origin master --force  # Use with caution!
   ```

4. **Verify Fix**
   - Wait for deployment
   - Test core functionality
   - Monitor logs

---

## 📚 Useful Resources

- **Render Docs**: https://docs.render.com/
- **Flask Docs**: https://flask.palletsprojects.com/
- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **CSS Best Practices**: https://web.dev/performance/
- **Optimization Tips**: https://web.dev/metrics/

---

## ✅ Deployment Checklist

Before each deployment:
- [ ] All features tested locally
- [ ] No console errors
- [ ] CSS changes verified
- [ ] Commit messages clear
- [ ] `.env` not committed
- [ ] Dependencies updated in requirements.txt
- [ ] No sensitive data in code
- [ ] Previous version working (for rollback)

---

## 📞 Getting Help

### Common Issues
1. **"Service idle" error**: Normal on free tier, will wake up automatically
2. **"502 Bad Gateway"**: App crashed, check Render logs
3. **"503 Service Unavailable"**: Deploying or building, wait 5 minutes
4. **Old version showing**: Hard refresh browser cache

### Debug Mode
1. Set `debug=True` in Flask (local only)
2. Check `console.log()` in browser DevTools
3. Use Network tab to inspect API calls
4. Check Render logs for backend errors

---

**Last Updated**: 2025-01-31  
**Version**: 1.0  
**Status**: Ready for Production
