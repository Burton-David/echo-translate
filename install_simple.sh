#!/bin/bash
set -e

echo "Installing EchoTranslate..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create wrapper that activates conda and runs from project directory
cat > /tmp/echotranslate << EOF
#!/bin/bash
source ~/miniconda3/bin/activate echotranslate 2>/dev/null || source ~/anaconda3/bin/activate echotranslate 2>/dev/null
cd "$SCRIPT_DIR"
python echotranslate "\$@"
EOF

# Install to /usr/local/bin
chmod +x /tmp/echotranslate
sudo mv /tmp/echotranslate /usr/local/bin/

echo "Installation complete!"
echo ""
echo "You can now run: echotranslate"
echo ""
echo "This will open a menu where you can:"
echo "  - Create voice profiles"
echo "  - Practice pronunciation with translations"
echo "  - View saved practice sessions"
echo ""
echo "To uninstall: sudo rm /usr/local/bin/echotranslate"