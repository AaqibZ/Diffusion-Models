# Web Interface Deployment Guide

## 🚀 How to Use the Web Interface

The web interface (`diffusion_recommender_web.html`) is a **standalone HTML file** that runs entirely in your browser with **NO server or Python interpreter needed**!

---

## ✅ Option 1: Open Directly (Simplest)

1. **Download** the file `diffusion_recommender_web.html`
2. **Double-click** the file
3. It will open in your default web browser
4. **Done!** Start answering questions

---

## ✅ Option 2: Host on a Web Server

### Using Python's Built-in Server

```bash
# Navigate to the folder containing the HTML file
cd /path/to/folder

# Start a simple HTTP server (Python 3)
python -m http.server 8000

# Or with Python 2
python -m SimpleHTTPServer 8000

# Open browser to: http://localhost:8000/diffusion_recommender_web.html
```

### Using Node.js

```bash
# Install http-server globally
npm install -g http-server

# Run server
http-server -p 8000

# Open browser to: http://localhost:8000/diffusion_recommender_web.html
```

### Using VS Code Live Server

1. Install "Live Server" extension in VS Code
2. Right-click on `diffusion_recommender_web.html`
3. Select "Open with Live Server"

---

## ✅ Option 3: Deploy Online (Free Hosting)

### GitHub Pages

1. Create a GitHub repository
2. Upload `diffusion_recommender_web.html`
3. Rename it to `index.html`
4. Go to Settings → Pages
5. Select main branch as source
6. Your site will be at: `https://username.github.io/repo-name/`

### Netlify

1. Go to [netlify.com](https://www.netlify.com/)
2. Drag and drop the HTML file
3. Get instant URL like: `https://random-name.netlify.app/`

### Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Follow prompts - your site will be live in seconds!
```

### CodePen / JSFiddle

1. Copy HTML content
2. Paste into HTML section
3. Copy `<style>` into CSS section
4. Copy `<script>` into JS section
5. Share the link!

---

## 🎨 Features of the Web Interface

### ✨ No Dependencies
- Pure HTML/CSS/JavaScript
- No frameworks required
- No build process
- No installation needed

### 💾 Data Persistence
- Uses browser's `localStorage`
- Learning weights persist across sessions
- No backend database needed

### 📱 Responsive Design
- Works on desktop, tablet, mobile
- Touch-friendly buttons
- Adaptive layout

### 🎯 Full Feature Parity
- All 34 models from paper
- 13 comprehensive questions
- Multi-factor scoring algorithm
- Research-validated recommendations
- Export to JSON

### 🔒 Privacy-First
- Everything runs locally
- No data sent to servers
- No tracking
- No cookies (except localStorage for weights)

---

## 📊 How It Works

### Architecture

```
┌─────────────────────────────────────┐
│  Browser (Client-Side Only)        │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   HTML Structure             │  │
│  │   - Questions                │  │
│  │   - Results Display          │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   CSS Styling                │  │
│  │   - Responsive Layout        │  │
│  │   - Animations               │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   JavaScript Logic           │  │
│  │   - Model Database (34)      │  │
│  │   - Scoring Algorithm        │  │
│  │   - localStorage for weights │  │
│  │   - Export functionality     │  │
│  └──────────────────────────────┘  │
│                                     │
└─────────────────────────────────────┘
         ↓ No server needed!
```

### Data Flow

1. **User Input** → Questions answered
2. **Processing** → JavaScript scoring algorithm
3. **Ranking** → Multi-factor weighted scoring
4. **Display** → Top 7 recommendations
5. **Export** → JSON download (optional)
6. **Learning** → Weights saved to localStorage

---

## 🔧 Customization

### Change Colors

Edit the CSS in the `<style>` section:

```css
/* Main gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Change to your colors */
background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
```

### Add More Models

In the JavaScript section, add to `MODELS` object:

```javascript
"YourModel": {
    taxonomy: "Process-Explanatory",
    temporal: true,
    competitive: false,
    behavioral: true,
    submodular: true,
    monotone: true,
    complexity: "Polynomial",
    best_for: ["your_use_case"],
    coverage: "high",
    temporal_accuracy: "very_high",
    description: "Your model description"
}
```

### Modify Questions

Edit the `questions` array:

```javascript
{
    id: 14,
    type: "radio",
    question: "Your question?",
    description: "Help text",
    options: [
        { value: "option1", label: "Option 1" },
        { value: "option2", label: "Option 2" }
    ]
}
```

---

## 🐛 Troubleshooting

### Issue: Page doesn't load
- **Solution**: Make sure you're opening the `.html` file, not a compressed version
- Check browser console (F12) for errors

### Issue: Styling looks broken
- **Solution**: Ensure the entire file is intact
- Try a different browser

### Issue: localStorage not persisting
- **Solution**: 
  - Check if you're in private/incognito mode (localStorage disabled)
  - Some browsers block localStorage for local files - use a web server

### Issue: Export doesn't work
- **Solution**: 
  - Check browser permissions for downloads
  - Try right-click → Save As instead

---

## 📱 Mobile Optimization

The interface is fully mobile-responsive:

- **Portrait mode**: Single column layout
- **Landscape mode**: Optimized for wider screens
- **Touch targets**: Large buttons (min 44px)
- **No hover states**: Touch-friendly interactions

---

## 🔐 Security Notes

### Safe to Use
✅ No external dependencies  
✅ No remote code execution  
✅ No data transmission  
✅ No user tracking  

### localStorage Contents
The only data stored is:
```json
{
  "diffusion_model_weights": {
    "IC": 1.0,
    "LT": 1.2,
    "SI": 0.9,
    // ... other models
  }
}
```

To clear: Open browser console and run:
```javascript
localStorage.removeItem('diffusion_model_weights');
```

---

## 🎯 Performance

### Load Time
- **Initial load**: < 1 second
- **Question navigation**: Instant
- **Results calculation**: ~1.5 seconds (simulated for UX)
- **Export**: Instant

### Browser Compatibility

| Browser | Minimum Version | Notes |
|---------|----------------|-------|
| Chrome | 60+ | ✅ Fully supported |
| Firefox | 60+ | ✅ Fully supported |
| Safari | 12+ | ✅ Fully supported |
| Edge | 79+ | ✅ Fully supported |
| Opera | 47+ | ✅ Fully supported |
| IE | ❌ Not supported | Use modern browser |

---

## 📈 Comparison: Web vs Python

| Feature | Web Interface | Python CLI |
|---------|--------------|------------|
| **Installation** | None | Python required |
| **Startup** | Instant | ~1 second |
| **User Experience** | Visual, interactive | Text-based |
| **Mobile Support** | ✅ Yes | ❌ No |
| **Sharing** | Share URL | Share .py file |
| **Deployment** | Drag & drop | Setup required |
| **Updates** | Edit HTML | Edit Python |
| **Accessibility** | Better | Good |
| **Learning** | localStorage | JSON file |

---

## 🎓 Advanced Usage

### Embed in Website

```html
<iframe 
    src="diffusion_recommender_web.html" 
    width="100%" 
    height="800px" 
    frameborder="0">
</iframe>
```

### Use as Chrome Extension

1. Create `manifest.json`:
```json
{
  "manifest_version": 3,
  "name": "Diffusion Model Recommender",
  "version": "1.0",
  "action": {
    "default_popup": "diffusion_recommender_web.html"
  }
}
```

2. Load unpacked extension in Chrome
3. Click extension icon to use

### Progressive Web App (PWA)

Add this to `<head>`:

```html
<link rel="manifest" href="manifest.json">
<meta name="theme-color" content="#667eea">

<script>
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js');
}
</script>
```

---

## 📚 Educational Use

Perfect for:
- **Research presentations** - Live demo in browser
- **Teaching** - No setup barrier for students
- **Workshops** - Everyone can participate instantly
- **Papers** - Include as supplementary material (single file)

---

## 💡 Pro Tips

1. **Bookmark it**: Add to browser bookmarks for quick access
2. **Save locally**: Keep a copy offline for presentations
3. **Version control**: Use Git to track customizations
4. **Share easily**: Upload to Dropbox/Google Drive, share link
5. **Print results**: Use browser print (Ctrl+P) to save PDF

---

## 🚀 Quick Start Checklist

- [ ] Download `diffusion_recommender_web.html`
- [ ] Double-click to open
- [ ] Answer 13 questions
- [ ] Get personalized recommendations
- [ ] Export results (optional)
- [ ] Share with colleagues!

---

## ❓ FAQ

**Q: Do I need internet?**  
A: No! Works completely offline after downloading.

**Q: Can I edit the questions?**  
A: Yes! Edit the `questions` array in the JavaScript section.

**Q: Will it work on my phone?**  
A: Yes! Fully responsive design.

**Q: Can I host this on my company intranet?**  
A: Absolutely! Just upload the single HTML file.

**Q: How do I reset the learning weights?**  
A: Open browser console: `localStorage.clear()`

**Q: Can I white-label this?**  
A: Yes! Edit the HTML/CSS to match your branding.

**Q: Is the code minified?**  
A: No, it's fully readable for transparency and customization.

**Q: Can I use this commercially?**  
A: Check the license. The recommendation logic is based on published research.

---

## 📧 Support

If you encounter issues:
1. Check browser console (F12) for errors
2. Try a different browser
3. Clear localStorage and retry
4. Open an issue on GitHub (if available)

---

## 🎉 You're Ready!

Just double-click the HTML file and start recommending diffusion models!

**No installation. No configuration. No hassle.**

---

**Last Updated:** January 2026  
**Tested On:** Chrome 120+, Firefox 120+, Safari 17+, Edge 120+  
**File Size:** ~58KB (single file)  
**Dependencies:** None ✨
