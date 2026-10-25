//
//  match.h
//  Efficient Compression Tool
//
//  Created by Felix Hanau on 31.12.15.
//  Copyright © 2015 Felix Hanau.
//

/*
 Finds how long the match of scan and match is. Can be used to find how many
 bytes starting from scan, and from match, are equal. Returns the last byte
 after scan, which is still equal to the correspondinb byte after match.
 scan is the position to compare
 match is the earlier position to compare.
 end is the last possible byte, beyond which to stop looking.
 safe_end is a few (8) bytes before end, for comparing multiple bytes at once.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
  #define _RESTRICT __restrict__
#elif defined(_MSC_VER)
  #define _RESTRICT __restrict
#else
  #define _RESTRICT
#endif

static inline uint64_t LoadMatchWord(const unsigned char* address) {
  uint64_t word;
  memcpy(&word, address, sizeof(word));
  return word;
}

#ifdef __GNUC__
static inline unsigned FirstDifferentByte(uint64_t difference) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return (unsigned)(__builtin_clzll(difference) >> 3);
#else
  return (unsigned)(__builtin_ctzll(difference) >> 3);
#endif
}
#endif

#ifdef __GNUC__
__attribute__ ((always_inline, hot))
#elif defined(_MSC_VER)
__forceinline
#endif
static inline const unsigned char* GetMatch(const unsigned char* _RESTRICT scan,
                                            const unsigned char* _RESTRICT match,
                                            const unsigned char* _RESTRICT end
                                            , const unsigned char* safe_end) {
#undef _RESTRICT
#ifdef __GNUC__
  /* Optimized Function based on cloudflare's zlib fork.
   * Note that this may read up to 15 bytes beyond end,
   * so it is necessary to allocate memory beyond the
   * end of the data whenever ZopfliDeflate is called.
   */
  do {
    uint64_t sv = LoadMatchWord(scan);
    uint64_t mv = LoadMatchWord(match);
    uint64_t xor = sv ^ mv;
    if (xor) {
      scan += FirstDifferentByte(xor);
      break;
    }
    scan += 8;
    match += 8;
    sv = LoadMatchWord(scan);
    mv = LoadMatchWord(match);
    xor = sv ^ mv;
    if (xor) {
      scan += FirstDifferentByte(xor);
      break;
    }
    scan += 8;
    match += 8;
  } while (scan < end);

  if (unlikely(scan > end))
    scan = end;
  return scan;

#else

  if (sizeof(size_t) == 8) {
    /* 8 checks at once per array bounds check (size_t is 64-bit). */
    while (scan < safe_end && memcmp(scan, match, sizeof(size_t)) == 0) {
      scan += 8;
      match += 8;
    }
  } else if (sizeof(unsigned) == 4) {
    /* 4 checks at once per array bounds check (unsigned is 32-bit). */
    while (scan < safe_end
           && memcmp(scan, match, sizeof(unsigned)) == 0) {
      scan += 4;
      match += 4;
    }
  } else {
    /* do 8 checks at once per array bounds check. */
    while (scan < safe_end && *scan == *match && *++scan == *++match
           && *++scan == *++match && *++scan == *++match
           && *++scan == *++match && *++scan == *++match
           && *++scan == *++match && *++scan == *++match) {
      scan++; match++;
    }
  }
  /* The remaining few bytes. */
  while (scan != end && *scan == *match) {
    scan++; match++;
  }
  return scan;
#endif
}
