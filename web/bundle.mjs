import { readFile } from "node:fs/promises";
import { build } from "esbuild";
import { fileURLToPath } from "node:url";

const destination = process.argv[2];
const packageRoot = fileURLToPath(new URL(".", import.meta.url));

const vp9Alpha = {
    name: "webm-alpha",
    setup(builder) {
        builder.onLoad({ filter: /webm-muxer\.mjs$/ }, async ({ path }) => {
            let contents = await readFile(path, "utf8");
            // webm-muxer already writes BlockAdditions for other tracks. Supply VP9 alpha
            // as BlockAdditional (ID 1) on the corresponding video block.
            const chunk = '    if (__privateGet(this, _options).video.codec === "V_VP9")';
            const block = '      chunk.additions ? { id: 30113 /* BlockAdditions */, data: chunk.additions } : null';
            if (!contents.includes(chunk) || !contents.includes(block)) {
                throw new Error("The pinned webm-muxer source changed. Review the VP9 alpha integration.");
            }
            contents = contents.replace(chunk, `    if (meta?.zoidcraftAlpha) {
      videoChunk.additions = [{ id: 166, data: [
        { id: 238, data: 1 },
        { id: 165, data: meta.zoidcraftAlpha }
      ] }];
      if (type === "delta" && this.zoidcraftPreviousTimestamp !== undefined) {
        let reference = new Uint8Array(4);
        new DataView(reference.buffer).setInt32(0, Math.floor((this.zoidcraftPreviousTimestamp - timestamp) / 1000));
        videoChunk.reference = { id: 251, data: reference };
      }
      this.zoidcraftPreviousTimestamp = timestamp;
    }
${chunk}`).replace(block, `${block},\n      chunk.reference ?? null`);
            return { contents, loader: "js" };
        });
    },
};

for (const [name, names] of Object.entries({
    mp4box: "createFile, DataStream, MP4BoxBuffer",
    "mp4-muxer": "ArrayBufferTarget, Muxer",
    "webm-muxer": "ArrayBufferTarget, Muxer",
    fflate: "Zip, ZipPassThrough",
})) {
    await build({
        stdin: { contents: `export { ${names} } from "${name}";`, resolveDir: packageRoot },
        bundle: true,
        minify: true,
        format: "esm",
        platform: "browser",
        target: "es2022",
        legalComments: "inline",
        plugins: name === "webm-muxer" ? [vp9Alpha] : [],
        outfile: `${destination}/${name}.js`,
    });
}
