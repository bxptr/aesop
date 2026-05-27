#include <sys/timers.h>
#include <ti/getcsc.h>
#include <ti/screen.h>
#include <ti/ui.h>

#include "../include/aesop_runtime.h"

static page_history_t g_page_history;

static void run_generated_model(void)
{
    load_text_rng_state();

    for (;;)
    {
        uint8_t restart = 0;
        uint16_t input_id;
        uint8_t action;

        input_id = token_start_story(1);
        page_history_reset(&g_page_history);

        while (!restart)
        {
            uint16_t screen_pos = 0;
            uint32_t start;
            uint32_t end;
            int24_t checksum = 0;
            uint8_t abort_key = 0;
            gen_result_t result = {0};

            result.magic[0] = 'G';
            result.magic[1] = 'E';
            result.magic[2] = 'N';
            result.magic[3] = 'R';
            result.version = 2;
            result.h = TOKEN_HIDDEN_SIZE;
            result.vocab = TOKEN_VOCAB_SIZE;

            os_ClrHome();
            if (input_id == TOKEN_EOS_ID)
            {
                input_id = token_start_story(1);
                os_ClrHome();
            }
            timer_Set(1, 0);
            timer_Enable(1, TIMER_CPU, TIMER_NOINT, TIMER_UP);
            start = timer_GetSafe(1, TIMER_UP);

            while (screen_pos < GENERATE_TOKENS)
            {
                abort_key = poll_exit_key();
                if (abort_key != 0)
                {
                    break;
                }
                if (input_id == TOKEN_EOS_ID)
                {
                    break;
                }

                token_emit_text(input_id, &result, &screen_pos);
                input_id = token_rnn_step(input_id, &checksum);
            }

            end = timer_GetSafe(1, TIMER_UP);
            timer_Disable(1);

            if (abort_key != 0)
            {
                save_text_rng_state();
                os_RunIndicOff();
                return;
            }

            result.tokens = screen_pos;
            result.text[screen_pos] = '\0';
            result.cycles_generate = end - start;
            result.last_id = input_id;
            write_gen_results(&result);
            page_history_push(&g_page_history, &result);
            save_text_rng_state();
            os_RunIndicOff();

            action = wait_for_story_action();
            for (;;)
            {
                if (action == sk_Clear || action == sk_Mode)
                {
                    return;
                }
                if (action == sk_Enter)
                {
                    page_history_reset(&g_page_history);
                    os_ClrHome();
                    input_id = token_start_story(0);
                    break;
                }
                if (action == sk_Left)
                {
                    if (page_history_back(&g_page_history, &action))
                    {
                        continue;
                    }
                    action = wait_for_story_action();
                    continue;
                }
                if (action == sk_Right && page_history_forward(&g_page_history, &action))
                {
                    continue;
                }
                break;
            }
        }
    }
}

int main(void)
{
    run_generated_model();
    return 0;
}
