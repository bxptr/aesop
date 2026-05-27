#include <ti/getcsc.h>
#include <ti/screen.h>
#include <ti/ui.h>

#include "../include/aesop_runtime.h"

uint8_t wait_for_story_action(void)
{
    uint8_t key;

    while (os_GetCSC() != 0)
    {
    }

    for (;;)
    {
        key = os_GetCSC();
        if (key == sk_Right || key == sk_Left || key == sk_Enter || key == sk_Clear || key == sk_Mode)
        {
            while (os_GetCSC() != 0)
            {
            }
            return key;
        }
    }
}

uint8_t poll_exit_key(void)
{
    uint8_t key = os_GetCSC();

    if (key == sk_Clear || key == sk_Mode)
    {
        while (os_GetCSC() != 0)
        {
        }
        return key;
    }

    return 0;
}

void display_generated_char(uint16_t *screen_pos, char ch)
{
    uint8_t row;
    uint8_t col;
    char text[2];

    if (*screen_pos >= DISPLAY_COLS * DISPLAY_ROWS)
    {
        return;
    }

    if (ch == '\n')
    {
        ch = ' ';
    }

    row = (uint8_t)(*screen_pos / DISPLAY_COLS);
    col = (uint8_t)(*screen_pos % DISPLAY_COLS);
    text[0] = ch;
    text[1] = '\0';
    os_SetCursorPos(row, col);
    os_PutStrFull(text);
    (*screen_pos)++;
}

void display_loading_progress(uint8_t step, uint8_t total)
{
    uint8_t i;
    uint8_t filled;
    char cell[2] = {' ', '\0'};
    const uint8_t bar_width = 5;
    const uint8_t bar_col = (uint8_t)((DISPLAY_COLS - bar_width) / 2);

    if (total == 0)
    {
        total = 1;
    }
    filled = (uint8_t)(((uint16_t)step * bar_width) / total);
    if (filled > bar_width)
    {
        filled = bar_width;
    }

    os_SetCursorPos(3, 9);
    os_PutStrFull("Loading");
    os_SetCursorPos(4, bar_col);
    os_PutStrFull("-----");
    cell[0] = '#';
    for (i = 0; i < filled; i++)
    {
        os_SetCursorPos(4, (uint8_t)(bar_col + i));
        os_PutStrFull(cell);
    }
}

static void copy_gen_result(gen_result_t *dst, const gen_result_t *src)
{
    uint16_t i;
    const uint8_t *src_bytes = (const uint8_t *)src;
    uint8_t *dst_bytes = (uint8_t *)dst;

    for (i = 0; i < sizeof(*dst); i++)
    {
        dst_bytes[i] = src_bytes[i];
    }
}

void page_history_reset(page_history_t *history)
{
    history->count = 0;
    history->pos = 0;
}

void page_history_push(page_history_t *history, const gen_result_t *result)
{
    uint8_t i;

    if (history->count < PAGE_HISTORY_COUNT)
    {
        copy_gen_result(&history->pages[history->count], result);
        history->pos = history->count;
        history->count++;
        return;
    }

    for (i = 1; i < PAGE_HISTORY_COUNT; i++)
    {
        copy_gen_result(&history->pages[i - 1], &history->pages[i]);
    }
    copy_gen_result(&history->pages[PAGE_HISTORY_COUNT - 1], result);
    history->pos = PAGE_HISTORY_COUNT - 1;
}

static uint8_t display_history_page(page_history_t *history)
{
    uint16_t i;
    uint16_t screen_pos = 0;
    const gen_result_t *result = &history->pages[history->pos];

    os_ClrHome();
    for (i = 0; i < result->tokens && i < GENERATE_TOKENS; i++)
    {
        display_generated_char(&screen_pos, result->text[i]);
    }
    write_gen_results(result);
    os_RunIndicOff();
    return wait_for_story_action();
}

uint8_t page_history_back(page_history_t *history, uint8_t *action)
{
    if (history->pos == 0)
    {
        return 0;
    }

    history->pos--;
    *action = display_history_page(history);
    return 1;
}

uint8_t page_history_forward(page_history_t *history, uint8_t *action)
{
    if (history->pos + 1 >= history->count)
    {
        return 0;
    }

    history->pos++;
    *action = display_history_page(history);
    return 1;
}
