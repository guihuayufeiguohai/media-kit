SPACE := $(subst ,, )
HEADERS_DESTDIR_ESCAPED=$(subst $(SPACE),\ ,${HEADERS_DESTDIR})

all: ${HEADERS_DESTDIR_ESCAPED}/mpv/*.h

# The headers MUST match the libmpv that Mpv.framework actually ships:
# mpv 0.41 plus the mpv_kit Vulkan render API extension (render_vk.h).
#
# Upstream mpv tarballs cannot provide that:
#   * v0.36.0 (what this Makefile used to download) has no render_vk.h and
#     no MPV_RENDER_API_TYPE_VULKAN at all;
#   * its layout is libmpv/*.h while 0.41 uses include/mpv/*.h.
# So take them straight from the published Vulkan bundle, which is built from
# the same source and installs exactly the headers libmpv was compiled with.
#
# Bundle layout: ./include/mpv/{client,render,render_gl,render_vk,stream_cb}.h
MPV_BUNDLE_VERSION=v0.1.0-vulkan.2
MPV_BUNDLE_SHA256SUM=e13c23dca0126cfe4d05178df022e2506485130dc5f2af60a505aec27fd7376d
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

.cache/headers/bundle.tar.gz:
	mkdir -p .cache/headers
	rm -f .cache/headers/*.tmp .cache/headers/bundle.tar.gz
	@set -e; \
	origin="https://github.com/${MPV_BUNDLE_PATH}"; \
	urls="$$origin"; \
	for m in $(MPV_BUNDLE_MIRRORS); do urls="$$urls $${m}$$origin"; done; \
	for u in $$urls; do \
		echo "==> headers: trying $$u"; \
		if curl -fL --retry 2 --retry-delay 1 --connect-timeout 20 -o .cache/headers/bundle.tar.gz.tmp "$$u"; then \
			echo "==> headers: ok"; exit 0; \
		fi; \
	done; \
	echo "==> headers: every source failed; retry with MPV_BUNDLE_MIRRORS=<prefix>"; \
	exit 1
	echo "${MPV_BUNDLE_SHA256SUM}  .cache/headers/bundle.tar.gz.tmp" | shasum -a 256 -c -
	mv .cache/headers/bundle.tar.gz.tmp .cache/headers/bundle.tar.gz
	touch .cache/headers/bundle.tar.gz

${HEADERS_DESTDIR_ESCAPED}/mpv/*.h: .cache/headers/bundle.tar.gz
	rm -rf .cache/headers/extract
	mkdir -p .cache/headers/extract
	# mpv/ for the API, vulkan/ + vk_video/ because render_vk.h includes
	# <vulkan/vulkan.h>. vulkan_core.h then includes "vk_video/..." by a
	# RELATIVE path, so vk_video must stay a sibling of vulkan/ and their
	# common parent must be on the include path -- that is exactly what
	# Headers/ becomes inside media_kit_video.framework.
	${TAR} -xvf .cache/headers/bundle.tar.gz -C .cache/headers/extract ./include/mpv ./include/vulkan ./include/vk_video
	# Wipe the destination first: an earlier run may have copied the full
	# Vulkan tree in, and leftover platform headers would still be picked up
	# by source_files and pulled into the umbrella.
	rm -rf ${HEADERS_DESTDIR_ESCAPED}/vulkan ${HEADERS_DESTDIR_ESCAPED}/vk_video
	mkdir -p ${HEADERS_DESTDIR_ESCAPED}/mpv ${HEADERS_DESTDIR_ESCAPED}/vulkan ${HEADERS_DESTDIR_ESCAPED}/vk_video
	cp .cache/headers/extract/include/mpv/*.h ${HEADERS_DESTDIR_ESCAPED}/mpv/
	# Only the headers iOS can actually compile. CocoaPods generates an
	# umbrella header that imports every public header directly, bypassing
	# the `#ifdef VK_USE_PLATFORM_*` guards in vulkan.h -- so copying the
	# whole tree made the umbrella pull in vulkan_fuchsia.h (zx_handle_t),
	# vulkan_win32.h (HWND/HINSTANCE) and vulkan_ggp.h (GgpStreamDescriptor),
	# none of which exist on iOS.
	#
	# vulkan.h only includes vk_platform.h and vulkan_core.h unconditionally;
	# every other include is guarded, and the only guard we define is
	# VK_USE_PLATFORM_METAL_EXT (see the podspec), so vulkan_metal.h is the
	# one platform header needed. It is self-contained: it typedefs
	# IOSurfaceRef itself and falls back to `void*` for the MTL handles.
	cp .cache/headers/extract/include/vulkan/vulkan.h \
	   .cache/headers/extract/include/vulkan/vulkan_core.h \
	   .cache/headers/extract/include/vulkan/vk_platform.h \
	   .cache/headers/extract/include/vulkan/vulkan_metal.h \
	   .cache/headers/extract/include/vulkan/vulkan_ios.h \
	   ${HEADERS_DESTDIR_ESCAPED}/vulkan/
	cp .cache/headers/extract/include/vk_video/*.h ${HEADERS_DESTDIR_ESCAPED}/vk_video/
	# Fail loudly: a silent 0.36-style header set, or a missing Vulkan tree,
	# would only surface as a module-build error deep inside the host app.
	test -f ${HEADERS_DESTDIR_ESCAPED}/mpv/render_vk.h
	test -f ${HEADERS_DESTDIR_ESCAPED}/mpv/client.h
	test -f ${HEADERS_DESTDIR_ESCAPED}/vulkan/vulkan_core.h
	test -f ${HEADERS_DESTDIR_ESCAPED}/vulkan/vulkan_metal.h
	test -f ${HEADERS_DESTDIR_ESCAPED}/vk_video/vulkan_video_codecs_common.h
	touch ${HEADERS_DESTDIR_ESCAPED}/mpv/*.h

.PHONY: clean
clean:
	rm -rf .cache
	rm -rf ${HEADERS_DESTDIR_ESCAPED}/mpv ${HEADERS_DESTDIR_ESCAPED}/vulkan ${HEADERS_DESTDIR_ESCAPED}/vk_video
