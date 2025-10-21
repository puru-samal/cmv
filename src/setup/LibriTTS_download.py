import asyncio
import hashlib
import os
import time
from typing import AsyncIterator, Tuple
import aiofiles
import httpx
from collections import namedtuple
import hydra
from omegaconf import OmegaConf, DictConfig
import logging
Args = namedtuple('Args', ['url', 'path', 'checksum'])

# Set up logging
log = logging.getLogger(__name__)

async def async_get(client: httpx.AsyncClient, url: str, chunk_size: int) -> AsyncIterator[bytes]:
    '''Async generator to fetch data in chunks from a URL.'''
    async with client.stream("GET", url=url) as response:
        # will raise httpx.HTTPStatusError for non-2xx if called
        response.raise_for_status()
        async for chunk in response.aiter_bytes(chunk_size=chunk_size):
            yield chunk
            
async def async_download_file(client: httpx.AsyncClient, sem: asyncio.Semaphore, args: Args, chunk_size: int) -> Tuple[Args, bool]:
    '''Download a file asynchronously and verify its checksum.
     Returns (Args, success: bool). Useful for tracking which files failed for retries.
    '''
    async with sem:  # limit concurrent downloads
        try:
            md5 = hashlib.md5()
            tmp_path = args.path + ".tmp"
            
            # ensure parent dir exists
            os.makedirs(os.path.dirname(tmp_path) or ".", exist_ok=True)

            # download/write file in chunks
            async with aiofiles.open(tmp_path, 'wb') as f:
                async for chunk in async_get(client, args.url, chunk_size):
                    await f.write(chunk)
                    md5.update(chunk)
            
            # verify checksum
            if md5.hexdigest() != args.checksum:
                log.error(f"[ERROR] Checksum mismatch for {args.path}. Expected {args.checksum}, got {md5.hexdigest()}")
                os.remove(tmp_path)
                return args, False

            log.info(f"[OK] {args.path}")
            os.replace(tmp_path, args.path)
            return args, True
        
        except Exception as e:
            log.error(f"[ERROR] {args.path} download failed: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return args, False

async def async_download(cfg: DictConfig) -> None:
    '''Main async function to download multiple files concurrently.'''

    # Config parameters
    download_dir = cfg.download_dir
    concurrency = cfg.concurrency
    num_retries = cfg.num_retries
    chunk_size = cfg.chunk_size
    links = OmegaConf.to_container(cfg.download_links, resolve=True) # avoid mutating input

    log.info(f"(Async) Downloading LibriTTS dataset files...")
    log.info(f"\t- Download directory: {download_dir}")
    log.info(f"\t- Concurrency: {concurrency}")
    log.info(f"\t- Max retries per file: {num_retries}")

    start_time = time.perf_counter()
    num_attempt = 0
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency*2)
    total_files = len(links)
    successes = 0
    
    # retry loop for failed downloads
    while num_attempt < num_retries and links:
        if num_attempt > 1:
            log.info(f"Retry attempt {num_attempt}/{num_retries} for {len(links)} failed downloads...")
        async with httpx.AsyncClient(limits=limits, http2=True, timeout=60.0) as client:
            tasks = []
            for path, info in links.items():
                dest = os.path.join(download_dir, path)
                tasks.append(async_download_file(client, sem, Args(url=info['url'], path=dest, checksum=info['checksum']), chunk_size))
            results = await asyncio.gather(*tasks, return_exceptions=False) 
            for args, success in results:
                if success:
                    del links[os.path.basename(args.path)]
                    successes += 1
        num_attempt += 1

    elapsed = time.perf_counter() - start_time
    log.info(f"{successes}/{total_files} succeeded: Took {elapsed:.2f} seconds.")
    if links:
        log.error(f"[ERROR] The following files failed to download after {num_retries} attempts:")
        for path in links.keys():
            log.error(f"\t- {path}")

@hydra.main(version_base=None, config_path="../../configs/setup", config_name="LibriTTS_download")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_download(cfg))

if __name__ == "__main__":
    main()
