all: Frameworks/*.xcframework Frameworks/.symlinks

# Vulkan build of libmpv (FFmpeg 8.1.3 + MoltenVK + mpv_kit's render_vk),
# published by guihuayufeiguohai/libmpv-darwin-build-upgrade. The bundle ships
# ./xcframework/*.xcframework: Mpv plus its whole runtime closure (Avcodec,
# Avformat, Avutil, Avfilter, Swresample, Swscale, Ass, Uchardet, Harfbuzz,
# Fribidi, Freetype, Png16, Mbedtls*, Dav1d, Xml2). Every dependency was
# rewritten to @rpath/<Name>.framework/<Name>, so vendoring all of them is
# what makes the app launch without dyld errors.
MPV_BUNDLE_VERSION=v0.1.0-vulkan.1
MPV_BUNDLE_SHA256SUM=06c9b1007619c422207728f88fc516d82e626b297e5737d2245f756dbb215d16
MPV_BUNDLE_PATH=guihuayufeiguohai/libmpv-darwin-build-upgrade/releases/download/${MPV_BUNDLE_VERSION}/vulkan-bundle-ios-arm64.tar.gz

# Direct github.com comes FIRST on purpose. Release assets are served by
# objects.githubusercontent.com, which stays reachable on networks where the
# github.com git endpoint is not; measured 4-10 MB/s direct versus ~390 KB/s
# through the best mirror. The mirrors below are only a fallback for networks
# that block the release CDN as well. Override with
#   make MPV_BUNDLE_MIRRORS="https://gh-proxy.com/"      (single mirror)
#   make MPV_BUNDLE_MIRRORS=                             (direct only)
MPV_BUNDLE_MIRRORS ?= https://gh-proxy.com/ https://ghproxy.net/ https://gh-proxy.org/

TAR=tar
TAR_WILDCARDS_FLAG := $(shell ${TAR} --version 2>&1 | grep -q 'GNU' && echo "--wildcards" || echo "")

SED=sed
SED_INPLACE_FLAG := $(shell ${SED} --version 2>&1 | grep -q 'GNU' && echo "" || echo "''")

.cache/bundle.tar.gz:
	mkdir -p .cache
	rm -f .cache/*.tmp .cache/bundle.tar.gz
	@set -e; \
	origin="https://github.com/${MPV_BUNDLE_PATH}"; \
	urls="$$origin"; \
	for m in $(MPV_BUNDLE_MIRRORS); do urls="$$urls $${m}$$origin"; done; \
	for u in $$urls; do \
		echo "==> bundle: trying $$u"; \
		if curl -fL --retry 2 --retry-delay 1 --connect-timeout 20 -o .cache/bundle.tar.gz.tmp "$$u"; then \
			echo "==> bundle: ok"; exit 0; \
		fi; \
	done; \
	echo "==> bundle: every source failed; retry with MPV_BUNDLE_MIRRORS=<prefix>"; \
	exit 1
	echo "${MPV_BUNDLE_SHA256SUM}  .cache/bundle.tar.gz.tmp" | shasum -a 256 -c -
	mv .cache/bundle.tar.gz.tmp .cache/bundle.tar.gz
	touch .cache/bundle.tar.gz

Frameworks/*.xcframework: .cache/bundle.tar.gz
	mkdir -p Frameworks
	rm -rf Frameworks/*.xcframework
	rm -rf .cache/extract
	mkdir -p .cache/extract
	${TAR} -xvf .cache/bundle.tar.gz -C .cache/extract ./xcframework
	cp -R .cache/extract/xcframework/*.xcframework Frameworks/
	# Mpv is the one media_kit_video links against; Uchardet/Avcodec/... are
	# its runtime dependencies. A missing Mpv means the bundle layout changed.
	test -d Frameworks/Mpv.xcframework
	test -d Frameworks/Uchardet.xcframework
	touch Frameworks/*.xcframework

Frameworks/.symlinks: Frameworks/*.xcframework
	rm -rf Frameworks/.symlinks
	mkdir -p Frameworks/.symlinks/mpv
	${SED} -i ${SED_INPLACE_FLAG} 's/\r$$//g' create_framework_symlinks.sh # remove CRLF line terminator added by Flutter packaging (https://github.com/media-kit/media-kit/issues/338)
	sh create_framework_symlinks.sh Frameworks/Mpv.xcframework Frameworks/.symlinks/mpv

.PHONY: clean
clean:
	rm -rf .cache
	rm -f ./Headers/mpv/*.h
	rm -rf ./Frameworks/*.xcframework
