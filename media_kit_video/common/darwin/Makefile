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
MPV_BUNDLE_VERSION=v0.1.0-vulkan.1
MPV_BUNDLE_SHA256SUM=06c9b1007619c422207728f88fc516d82e626b297e5737d2245f756dbb215d16
MPV_BUNDLE_URL=https://github.com/guihuayufeiguohai/libmpv-darwin-build-upgrade/releases/download/${MPV_BUNDLE_VERSION}/vulkan-bundle-ios-arm64.tar.gz

TAR=tar

.cache/headers/bundle.tar.gz:
	mkdir -p .cache/headers
	rm -f .cache/headers/*.tmp .cache/headers/bundle.tar.gz
	curl -L '${MPV_BUNDLE_URL}' -o .cache/headers/bundle.tar.gz.tmp
	shasum -a 256 -c <<< '${MPV_BUNDLE_SHA256SUM}  .cache/headers/bundle.tar.gz.tmp'
	mv .cache/headers/bundle.tar.gz.tmp .cache/headers/bundle.tar.gz
	touch .cache/headers/bundle.tar.gz

${HEADERS_DESTDIR_ESCAPED}/mpv/*.h: .cache/headers/bundle.tar.gz
	rm -rf .cache/headers/extract
	mkdir -p .cache/headers/extract
	${TAR} -xvf .cache/headers/bundle.tar.gz -C .cache/headers/extract ./include/mpv
	mkdir -p ${HEADERS_DESTDIR_ESCAPED}/mpv
	cp .cache/headers/extract/include/mpv/*.h ${HEADERS_DESTDIR_ESCAPED}/mpv/
	# Fail loudly: the Vulkan renderer needs these, and a silent 0.36-style
	# header set would only surface as a compile error deep in ObjC++.
	test -f ${HEADERS_DESTDIR_ESCAPED}/mpv/render_vk.h
	test -f ${HEADERS_DESTDIR_ESCAPED}/mpv/client.h
	touch ${HEADERS_DESTDIR_ESCAPED}/mpv/*.h

.PHONY: clean
clean:
	rm -rf .cache
