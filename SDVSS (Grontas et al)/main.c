// cab_sig_full.c (updated)
// - Uses clock_gettime() monotonic timing (RAW if available)
// - Prints OpenSSL version/build info once
// - Benchmarks CAB.Sign for b=1 (valid) AND b=0 (invalid signature generation path)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <openssl/ec.h>
#include <openssl/bn.h>
#include <openssl/sha.h>
#include <openssl/obj_mac.h>
#include <openssl/crypto.h>   // OpenSSL_version

#define ITERATIONS 200
#define MSG_LEN    200

#define CHECK(c,msg) do{ if(!(c)){ fprintf(stderr,"Error: %s\n", (msg)); exit(1);} }while(0)

// ---------------------------------------------------------------------
// High-resolution monotonic timer (Linux)
// Prefer CLOCK_MONOTONIC_RAW; fallback to CLOCK_MONOTONIC
// ---------------------------------------------------------------------
static double now_seconds(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
    CHECK(clock_gettime(CLOCK_MONOTONIC, &ts) == 0, "clock_gettime");
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------
// Random bytes (NOT timed)
// ---------------------------------------------------------------------
static void random_bytes(unsigned char *buf, size_t len) {
    FILE *fp = fopen("/dev/urandom", "rb");
    if (!fp) { perror("urandom"); exit(1); }
    if (fread(buf, 1, len, fp) != len) {
        perror("fread urandom");
        fclose(fp);
        exit(1);
    }
    fclose(fp);
}

// ---------------------------------------------------------------------
// Hash helper: H(m, x*) in Z_q
// SHA256(m || xcoord(x*)) mod q
// ---------------------------------------------------------------------
static int hash_m_xstar(const unsigned char *msg,  size_t msg_len,
                        const EC_POINT *x_star,
                        const EC_GROUP *group,
                        const BIGNUM *order,
                        BIGNUM *out,
                        BN_CTX *ctx) {
    int ret = 0;
    BIGNUM *x = BN_new(); CHECK(x, "BN_new x");

    int field_bits  = EC_GROUP_get_degree(group);
    int field_bytes = (field_bits + 7) / 8;

    unsigned char *buf = OPENSSL_malloc(field_bytes);
    CHECK(buf, "OPENSSL_malloc");

    unsigned char hash[SHA256_DIGEST_LENGTH];

    if (!EC_POINT_get_affine_coordinates(group, x_star, x, NULL, ctx))
        goto end;
    if (BN_bn2binpad(x, buf, field_bytes) < 0)
        goto end;

    SHA256_CTX sha;
    SHA256_Init(&sha);
    SHA256_Update(&sha, msg, msg_len);
    SHA256_Update(&sha, buf, field_bytes);
    SHA256_Final(hash, &sha);

    BN_bin2bn(hash, SHA256_DIGEST_LENGTH, out);
    if (!BN_mod(out, out, order, ctx)) goto end;

    ret = 1;

end:
    OPENSSL_free(buf);
    BN_free(x);
    return ret;
}

// ---------------------------------------------------------------------
// Keys / signature structs
// ---------------------------------------------------------------------
typedef struct {
    BIGNUM *s1, *s2;   // signer secret
    BIGNUM *s;         // verifier secret

    EC_POINT *g1;
    EC_POINT *g2;
    EC_POINT *v;       // pkS
    EC_POINT *k;       // pkV
} cab_keys_t;

typedef struct {
    EC_POINT *x_star;  // x*
    BIGNUM   *e_star;  // e*
    EC_POINT *sigma1;  // σ1
    BIGNUM   *sigma2;  // σ2
} cab_signature_t;

// ---------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------
static void cab_keys_init(cab_keys_t *keys, const EC_GROUP *group) {
    keys->s1 = BN_new();
    keys->s2 = BN_new();
    keys->s  = BN_new();
    keys->g1 = EC_POINT_new(group);
    keys->g2 = EC_POINT_new(group);
    keys->v  = EC_POINT_new(group);
    keys->k  = EC_POINT_new(group);
    CHECK(keys->s1 && keys->s2 && keys->s &&
          keys->g1 && keys->g2 && keys->v && keys->k,
          "alloc keys");
}

static void cab_keys_free(cab_keys_t *keys) {
    if (keys->s1) BN_free(keys->s1);
    if (keys->s2) BN_free(keys->s2);
    if (keys->s)  BN_free(keys->s);
    if (keys->g1) EC_POINT_free(keys->g1);
    if (keys->g2) EC_POINT_free(keys->g2);
    if (keys->v)  EC_POINT_free(keys->v);
    if (keys->k)  EC_POINT_free(keys->k);
}

static void cab_sig_init(cab_signature_t *sig, const EC_GROUP *group) {
    sig->x_star = EC_POINT_new(group);
    sig->e_star = BN_new();
    sig->sigma1 = EC_POINT_new(group);
    sig->sigma2 = BN_new();
    CHECK(sig->x_star && sig->e_star && sig->sigma1 && sig->sigma2,
          "alloc sig");
}

static void cab_sig_free(cab_signature_t *sig) {
    if (sig->x_star) EC_POINT_free(sig->x_star);
    if (sig->e_star) BN_free(sig->e_star);
    if (sig->sigma1) EC_POINT_free(sig->sigma1);
    if (sig->sigma2) BN_free(sig->sigma2);
}

// ---------------------------------------------------------------------
// Setup keys: g1=G, g2=hG, secrets, v, k
// ---------------------------------------------------------------------
static void cab_setup_keys(const EC_GROUP *group, const BIGNUM *order,
                           cab_keys_t *keys, BN_CTX *ctx) {
    cab_keys_init(keys, group);

    const EC_POINT *G = EC_GROUP_get0_generator(group);
    CHECK(EC_POINT_copy(keys->g1, G) == 1, "g1=G");

    // g2 = h*G with h = SHA256("g2") mod q
    unsigned char hbuf[SHA256_DIGEST_LENGTH];
    const char label[] = "g2";

    SHA256_CTX sha;
    SHA256_Init(&sha);
    SHA256_Update(&sha, label, sizeof(label) - 1);
    SHA256_Final(hbuf, &sha);

    BIGNUM *h = BN_new(); CHECK(h, "BN_new h");
    BN_bin2bn(hbuf, sizeof(hbuf), h);
    BN_mod(h, h, order, ctx);
    if (BN_is_zero(h)) BN_one(h);

    CHECK(EC_POINT_mul(group, keys->g2, NULL, G, h, ctx) == 1, "g2=hG");

    // secrets
    BN_rand_range(keys->s1, order);
    BN_rand_range(keys->s2, order);
    BN_rand_range(keys->s,  order);

    // v = g1^{-s1} + g2^{-s2}
    BIGNUM *neg1 = BN_new(), *neg2 = BN_new();
    EC_POINT *t1 = EC_POINT_new(group), *t2 = EC_POINT_new(group);
    CHECK(neg1 && neg2 && t1 && t2, "alloc v parts");

    BN_sub(neg1, order, keys->s1);
    BN_sub(neg2, order, keys->s2);
    EC_POINT_mul(group, t1, NULL, keys->g1, neg1, ctx);
    EC_POINT_mul(group, t2, NULL, keys->g2, neg2, ctx);
    EC_POINT_add(group, keys->v, t1, t2, ctx);

    // k = g1^s
    EC_POINT_mul(group, keys->k, NULL, keys->g1, keys->s, ctx);

    BN_free(h);
    BN_free(neg1); BN_free(neg2);
    EC_POINT_free(t1); EC_POINT_free(t2);
}

// ---------------------------------------------------------------------
// CAB.Sign (simulated signer+user)
// b=1 => valid signature path
// b=0 => beta1 random point (invalid signature by design)
// ---------------------------------------------------------------------
static int cab_sign(const unsigned char *m, size_t mlen,
                    int b,
                    const EC_GROUP *group, const BIGNUM *order,
                    const cab_keys_t *keys, cab_signature_t *sig,
                    BN_CTX *ctx) {
    int ok = 0;

    BIGNUM *r1 = BN_new(), *r2 = BN_new();
    BIGNUM *u1 = BN_new(), *u2 = BN_new(), *d = BN_new();
    BIGNUM *e  = BN_new();
    BIGNUM *y1 = BN_new(), *y2 = BN_new();
    BIGNUM *tmp = BN_new();

    EC_POINT *x   = EC_POINT_new(group);
    EC_POINT *t1  = EC_POINT_new(group);
    EC_POINT *t2  = EC_POINT_new(group);
    EC_POINT *t3  = EC_POINT_new(group);

    EC_POINT *beta1 = EC_POINT_new(group);
    BIGNUM   *beta2 = BN_new();

    CHECK(r1 && r2 && u1 && u2 && d && e && y1 && y2 && tmp &&
          x && t1 && t2 && t3 && beta1 && beta2,
          "alloc sign");

    // Signer: x = g1^r1 + g2^r2
    BN_rand_range(r1, order);
    BN_rand_range(r2, order);

    EC_POINT_mul(group, t1, NULL, keys->g1, r1, ctx);
    EC_POINT_mul(group, t2, NULL, keys->g2, r2, ctx);
    EC_POINT_add(group, x, t1, t2, ctx);

    // User: x* = x + g1^u1 + g2^u2 + v^d
    BN_rand_range(u1, order);
    BN_rand_range(u2, order);
    BN_rand_range(d,  order);

    EC_POINT_mul(group, t1, NULL, keys->g1, u1, ctx);
    EC_POINT_mul(group, t2, NULL, keys->g2, u2, ctx);
    EC_POINT_mul(group, t3, NULL, keys->v,  d,  ctx);

    EC_POINT_copy(sig->x_star, x);
    EC_POINT_add(group, sig->x_star, sig->x_star, t1, ctx);
    EC_POINT_add(group, sig->x_star, sig->x_star, t2, ctx);
    EC_POINT_add(group, sig->x_star, sig->x_star, t3, ctx);

    // e* = H(m, x*)
    CHECK(hash_m_xstar(m, mlen, sig->x_star, group, order, sig->e_star, ctx),
          "hash e*");

    // e = e* - d
    BN_mod_sub(e, sig->e_star, d, order, ctx);

    // y1 = r1 + e*s1 ; y2 = r2 + e*s2
    BN_mod_mul(tmp, e, keys->s1, order, ctx);
    BN_mod_add(y1, r1, tmp, order, ctx);

    BN_mod_mul(tmp, e, keys->s2, order, ctx);
    BN_mod_add(y2, r2, tmp, order, ctx);

    // beta1, beta2
    if (b == 1) {
        EC_POINT_mul(group, beta1, NULL, keys->k, y1, ctx); // beta1 = k^{y1}
    } else {
        // beta1 <-R G (random point)
        BIGNUM *rr = BN_new(); CHECK(rr, "BN_new rr");
        BN_rand_range(rr, order);
        EC_POINT_mul(group, beta1, NULL, keys->g1, rr, ctx);
        BN_free(rr);
    }
    BN_copy(beta2, y2);

    // Unblind: sigma1 = beta1 + k^{u1} ; sigma2 = beta2 + u2
    EC_POINT_mul(group, t1, NULL, keys->k, u1, ctx);
    EC_POINT_add(group, sig->sigma1, beta1, t1, ctx);

    BN_mod_add(sig->sigma2, beta2, u2, order, ctx);

    ok = 1;

    BN_free(r1); BN_free(r2);
    BN_free(u1); BN_free(u2); BN_free(d);
    BN_free(e);
    BN_free(y1); BN_free(y2);
    BN_free(tmp);

    EC_POINT_free(x);
    EC_POINT_free(t1); EC_POINT_free(t2); EC_POINT_free(t3);
    EC_POINT_free(beta1);
    BN_free(beta2);

    return ok;
}

// ---------------------------------------------------------------------
// CAB.Verify
// ---------------------------------------------------------------------
static int cab_verify(const unsigned char *m, size_t mlen,
                      const EC_GROUP *group, const BIGNUM *order,
                      const cab_keys_t *keys, const cab_signature_t *sig,
                      BN_CTX *ctx) {
    int ok = 0;

    BIGNUM *e2   = BN_new();
    BIGNUM *sig2s = BN_new();
    BIGNUM *es   = BN_new();
    CHECK(e2 && sig2s && es, "alloc verify BN");

    CHECK(hash_m_xstar(m, mlen, sig->x_star, group, order, e2, ctx),
          "hash e2");

    if (BN_cmp(e2, sig->e_star) != 0) {
        ok = 0;
        goto end;
    }

    EC_POINT *lhs = EC_POINT_new(group);
    EC_POINT *rhs = EC_POINT_new(group);
    EC_POINT *t1  = EC_POINT_new(group);
    EC_POINT *t2  = EC_POINT_new(group);
    CHECK(lhs && rhs && t1 && t2, "alloc verify points");

    // lhs = (x*)^s
    EC_POINT_mul(group, lhs, NULL, sig->x_star, keys->s, ctx);

    // rhs = sigma1 + g2^{sigma2*s} + v^{e* * s}
    BN_mod_mul(sig2s, sig->sigma2, keys->s, order, ctx);
    BN_mod_mul(es,    sig->e_star, keys->s, order, ctx);

    EC_POINT_mul(group, t1, NULL, keys->g2, sig2s, ctx);
    EC_POINT_mul(group, t2, NULL, keys->v,  es,    ctx);

    EC_POINT_copy(rhs, sig->sigma1);
    EC_POINT_add(group, rhs, rhs, t1, ctx);
    EC_POINT_add(group, rhs, rhs, t2, ctx);

    ok = (EC_POINT_cmp(group, lhs, rhs, ctx) == 0);

    EC_POINT_free(lhs);
    EC_POINT_free(rhs);
    EC_POINT_free(t1);
    EC_POINT_free(t2);

end:
    BN_free(e2);
    BN_free(sig2s);
    BN_free(es);
    return ok;
}

// ---------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------
static void print_sizes(const EC_GROUP *group,
                        const BIGNUM *order,
                        const cab_keys_t *keys,
                        const cab_signature_t *sig,
                        BN_CTX *ctx) {
    int scalar_bytes = BN_num_bytes(order);
    int scalar_bits  = scalar_bytes * 8;

    size_t v_bytes = EC_POINT_point2oct(group, keys->v, POINT_CONVERSION_COMPRESSED, NULL, 0, ctx);
    size_t k_bytes = EC_POINT_point2oct(group, keys->k, POINT_CONVERSION_COMPRESSED, NULL, 0, ctx);
    size_t x_bytes = EC_POINT_point2oct(group, sig->x_star, POINT_CONVERSION_COMPRESSED, NULL, 0, ctx);
    size_t s1_bytes = EC_POINT_point2oct(group, sig->sigma1, POINT_CONVERSION_COMPRESSED, NULL, 0, ctx);

    size_t sig_bytes = x_bytes + scalar_bytes + s1_bytes + scalar_bytes;

    printf("  Key / Signature sizes (compressed points):\n");
    printf("    scalar size (Z_q):       %d bytes (%d bits)\n", scalar_bytes, scalar_bits);
    printf("    sk_S = (s1,s2):          %d bytes (%d bits)\n", 2*scalar_bytes, 2*scalar_bits);
    printf("    sk_V = s:                %d bytes (%d bits)\n", scalar_bytes, scalar_bits);
    printf("    pk_S = v (EC point):     %zu bytes (%zu bits)\n", v_bytes, v_bytes*8);
    printf("    pk_V = k (EC point):     %zu bytes (%zu bits)\n", k_bytes, k_bytes*8);

    printf("    CAB signature (x*, e*, sigma1, sigma2):\n");
    printf("      total:                 %zu bytes (%zu bits)\n", sig_bytes, sig_bytes*8);
    printf("      breakdown:\n");
    printf("        x*      (EC point):  %zu bytes\n", x_bytes);
    printf("        e*      (scalar):    %d bytes\n", scalar_bytes);
    printf("        sigma1  (EC point):  %zu bytes\n", s1_bytes);
    printf("        sigma2  (scalar):    %d bytes\n", scalar_bytes);
}

// ---------------------------------------------------------------------
// Benchmark per curve
// ---------------------------------------------------------------------
static void benchmark_curve(int nid, const char *name) {
    printf("=== Curve: %s (NID=%d) ===\n", name, nid);

    BN_CTX *ctx = BN_CTX_new(); CHECK(ctx, "BN_CTX_new");
    EC_GROUP *group = EC_GROUP_new_by_curve_name(nid);
    CHECK(group, "EC_GROUP_new_by_curve_name");

    BIGNUM *order = BN_new(); CHECK(order, "BN_new order");
    CHECK(EC_GROUP_get_order(group, order, ctx) == 1, "get_order");

    cab_keys_t keys;
    cab_setup_keys(group, order, &keys, ctx);

    cab_signature_t sig;
    cab_sig_init(&sig, group);

    unsigned char msg[MSG_LEN];

    // warm-up
    random_bytes(msg, MSG_LEN);
    CHECK(cab_sign(msg, MSG_LEN, 1, group, order, &keys, &sig, ctx), "warm sign");
    CHECK(cab_verify(msg, MSG_LEN, group, order, &keys, &sig, ctx), "warm verify");

    print_sizes(group, order, &keys, &sig, ctx);
    printf("\n");

    // ----------------------------
    // CAB.Sign benchmark (b=1)
    // ----------------------------
    double sign_valid_sum = 0.0;
    for (int i = 0; i < ITERATIONS; i++) {
        unsigned char m[MSG_LEN];
        random_bytes(m, MSG_LEN);

        double t0 = now_seconds();
        if (!cab_sign(m, MSG_LEN, 1, group, order, &keys, &sig, ctx)) {
            fprintf(stderr, "CAB.Sign(b=1) failed at %d\n", i);
            exit(1);
        }
        double t1 = now_seconds();
        sign_valid_sum += (t1 - t0);
    }
    double sign_valid_avg = sign_valid_sum / (double)ITERATIONS;
    double sign_valid_thr = (double)ITERATIONS / (sign_valid_sum > 0 ? sign_valid_sum : 1e-9);

    // ----------------------------
    // CAB.Sign benchmark (b=0)  <-- requested
    // ----------------------------
    double sign_invalid_sum = 0.0;
    for (int i = 0; i < ITERATIONS; i++) {
        unsigned char m[MSG_LEN];
        random_bytes(m, MSG_LEN);

        double t0 = now_seconds();
        if (!cab_sign(m, MSG_LEN, 0, group, order, &keys, &sig, ctx)) {
            fprintf(stderr, "CAB.Sign(b=0) failed at %d\n", i);
            exit(1);
        }
        double t1 = now_seconds();
        sign_invalid_sum += (t1 - t0);
    }
    double sign_invalid_avg = sign_invalid_sum / (double)ITERATIONS;
    double sign_invalid_thr = (double)ITERATIONS / (sign_invalid_sum > 0 ? sign_invalid_sum : 1e-9);

    // ----------------------------
    // CAB.Verify benchmark (valid)
    // ----------------------------
    double ver_sum = 0.0;
    for (int i = 0; i < ITERATIONS; i++) {
        unsigned char m[MSG_LEN];
        random_bytes(m, MSG_LEN);

        // not timed: fresh valid signature
        if (!cab_sign(m, MSG_LEN, 1, group, order, &keys, &sig, ctx)) {
            fprintf(stderr, "CAB.Sign (for verify) failed at %d\n", i);
            exit(1);
        }

        double t0 = now_seconds();
        if (!cab_verify(m, MSG_LEN, group, order, &keys, &sig, ctx)) {
            fprintf(stderr, "CAB.Verify failed at %d\n", i);
            exit(1);
        }
        double t1 = now_seconds();
        ver_sum += (t1 - t0);
    }
    double ver_avg = ver_sum / (double)ITERATIONS;
    double ver_thr = (double)ITERATIONS / (ver_sum > 0 ? ver_sum : 1e-9);

    // ----------------------------
    // (Optional but useful) Verify invalid signatures benchmark
    // Expect reject; we still time it.
    // ----------------------------
    double ver_inv_sum = 0.0;
    int rejected = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        unsigned char m[MSG_LEN];
        random_bytes(m, MSG_LEN);

        // not timed: produce invalid signature b=0
        if (!cab_sign(m, MSG_LEN, 0, group, order, &keys, &sig, ctx)) {
            fprintf(stderr, "CAB.Sign(b=0) (for invalid verify) failed at %d\n", i);
            exit(1);
        }

        double t0 = now_seconds();
        int v = cab_verify(m, MSG_LEN, group, order, &keys, &sig, ctx);
        double t1 = now_seconds();
        ver_inv_sum += (t1 - t0);

        if (v == 0) rejected++;
    }
    double ver_inv_avg = ver_inv_sum / (double)ITERATIONS;
    double ver_inv_thr = (double)ITERATIONS / (ver_inv_sum > 0 ? ver_inv_sum : 1e-9);

    // Print
    printf("  Algorithm CAB.Sign (b=1, valid, random 200B m):\n");
    printf("    avg time/op:   %.6f sec (%.3f µs)\n", sign_valid_avg, sign_valid_avg * 1e6);
    printf("    throughput:    %.2f ops/sec\n\n", sign_valid_thr);

    printf("  Algorithm CAB.Sign (b=0, invalid-gen, random 200B m):\n");
    printf("    avg time/op:   %.6f sec (%.3f µs)\n", sign_invalid_avg, sign_invalid_avg * 1e6);
    printf("    throughput:    %.2f ops/sec\n\n", sign_invalid_thr);

    printf("  Algorithm CAB.Verify (valid, random 200B m):\n");
    printf("    avg time/op:   %.6f sec (%.3f µs)\n", ver_avg, ver_avg * 1e6);
    printf("    throughput:    %.2f ops/sec\n\n", ver_thr);

    printf("  Algorithm CAB.Verify (invalid signatures b=0, expected reject):\n");
    printf("    avg time/op:   %.6f sec (%.3f µs)\n", ver_inv_avg, ver_inv_avg * 1e6);
    printf("    throughput:    %.2f ops/sec\n", ver_inv_thr);
    printf("    rejected:      %d / %d\n\n", rejected, ITERATIONS);

    cab_sig_free(&sig);
    cab_keys_free(&keys);
    BN_free(order);
    EC_GROUP_free(group);
    BN_CTX_free(ctx);
}

// ---------------------------------------------------------------------
// Print OpenSSL build/runtime info once (requested #4)
// ---------------------------------------------------------------------
static void print_openssl_info(void) {
    printf("=== OpenSSL runtime/build info ===\n");
    printf("  Version:     %s\n", OpenSSL_version(OPENSSL_VERSION));
    printf("  Built on:    %s\n", OpenSSL_version(OPENSSL_BUILT_ON));
    printf("  Compiler:    %s\n", OpenSSL_version(OPENSSL_CFLAGS));
    printf("  Platform:    %s\n", OpenSSL_version(OPENSSL_PLATFORM));
    printf("  OPENSSLDIR:  %s\n", OpenSSL_version(OPENSSL_DIR));
    printf("==================================\n\n");
}

int main(void) {
    print_openssl_info();

    int nids[] = {
        NID_X9_62_prime192v1,    // NIST P-192
        NID_secp224r1,           // NIST P-224
        NID_X9_62_prime256v1,    // NIST P-256
        NID_secp256k1,           // SECP256k1
        NID_secp384r1,           // NIST P-384
        NID_secp521r1,           // NIST P-521
        NID_brainpoolP256r1,     // Brainpool P256
        NID_brainpoolP384r1      // Brainpool P384
    };
    const char *names[] = {
        "prime192v1 (NIST P-192)",
        "secp224r1 (NIST P-224)",
        "prime256v1 (NIST P-256)",
        "secp256k1 (Bitcoin)",
        "secp384r1 (NIST P-384)",
        "secp521r1 (NIST P-521)",
        "brainpoolP256r1",
        "brainpoolP384r1"
    };

    int num = (int)(sizeof(nids)/sizeof(nids[0]));
    for (int i = 0; i < num; i++) {
        benchmark_curve(nids[i], names[i]);
    }
    return 0;
}
