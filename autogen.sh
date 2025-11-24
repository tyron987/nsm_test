#!/bin/sh
set -e

echo "Generating build system files..."

libtoolize --force --copy 2>/dev/null || true
aclocal
autoconf
automake --add-missing --copy

echo "Now run ./configure and then make."

