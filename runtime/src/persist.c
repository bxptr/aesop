#include <sys/rtc.h>
#include <ti/vars.h>

#include "../include/aesop_runtime.h"

static uint24_t g_rng_state = 0x51A7E3U;

static uint24_t text_read_u24_le(const uint8_t *p)
{
    return (uint24_t)p[0] | ((uint24_t)p[1] << 8) | ((uint24_t)p[2] << 16);
}

static void text_write_u24_le(uint8_t *p, uint24_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
}

void load_text_rng_state(void)
{
    int archived = 0;
    var_t *var = os_GetAppVarData(TEXT_RNG_APPVAR_NAME, &archived);

    if (var != NULL &&
        var->size >= 7 &&
        var->data[0] == 'T' &&
        var->data[1] == 'R' &&
        var->data[2] == 'N' &&
        var->data[3] == 'G')
    {
        g_rng_state = text_read_u24_le(var->data + 4);
    }
    else
    {
        g_rng_state = (uint24_t)(rtc_Time() ^ 0x51A7E3U);
    }

    if (g_rng_state == 0)
    {
        g_rng_state = 0x51A7E3U;
    }

    (void)archived;
}

void save_text_rng_state(void)
{
    var_t *var;
    int archived = 0;

    if (os_GetAppVarData(TEXT_RNG_APPVAR_NAME, &archived) != NULL && !archived)
    {
        os_DelAppVar(TEXT_RNG_APPVAR_NAME);
    }

    var = os_CreateAppVar(TEXT_RNG_APPVAR_NAME, 7);
    if (var != NULL)
    {
        var->data[0] = 'T';
        var->data[1] = 'R';
        var->data[2] = 'N';
        var->data[3] = 'G';
        text_write_u24_le(var->data + 4, g_rng_state);
    }
}

uint8_t rng8(void)
{
    uint24_t x = g_rng_state;

    x ^= (uint24_t)(x << 7);
    x ^= (uint24_t)(x >> 9);
    x ^= (uint24_t)(x << 8);
    g_rng_state = x;
    return (uint8_t)(x >> 8);
}

void mix_text_rng_entropy(void)
{
    uint24_t now = (uint24_t)rtc_Time();

    g_rng_state ^= now;
    g_rng_state ^= (uint24_t)(now << 7);
    g_rng_state ^= 0xA53C19U;
    if (g_rng_state == 0)
    {
        g_rng_state = 0x51A7E3U;
    }

    (void)rng8();
    (void)rng8();
    (void)rng8();
}

void write_gen_results(const gen_result_t *result)
{
    uint16_t i;
    var_t *var;
    int archived = 0;

    if (os_GetAppVarData(GEN_APPVAR_NAME, &archived) != NULL && !archived)
    {
        os_DelAppVar(GEN_APPVAR_NAME);
    }

    var = os_CreateAppVar(GEN_APPVAR_NAME, sizeof(*result));
    if (var != NULL)
    {
        const uint8_t *src = (const uint8_t *)result;
        for (i = 0; i < sizeof(*result); i++)
        {
            var->data[i] = src[i];
        }
    }
}
