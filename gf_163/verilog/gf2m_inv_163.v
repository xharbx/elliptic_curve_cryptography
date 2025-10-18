`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/17/2025 06:43:54 AM
// Design Name: 
// Module Name: gf2m_inv163
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module gf2m_inv163(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [162:0] a,
    output reg  [162:0] inv,
    output reg         done
);

    // --------------------------------------------------------
    // Intermediate registers
    // --------------------------------------------------------
    reg [162:0] b1, b2, b3, b5, b10, b20, b40, b80, b81, b162;

    // --------------------------------------------------------
    // Squarer interface
    // --------------------------------------------------------
    reg  [162:0] sqr_in;
    wire [162:0] sqr_out;

    squerer_163 U_SQR (
        .data_in(sqr_in),
        .data_out(sqr_out)
    );

    // --------------------------------------------------------
    // Multiplier interface (3-cycle pipeline)
    // --------------------------------------------------------
    reg  [162:0] mul_a, mul_b;
    wire [162:0] mul_res;

    gf2m_mult163 U_MULT (
        .clk(clk),
        .a(mul_a),
        .b(mul_b),
        .c(mul_res)
    );

    // --------------------------------------------------------
    // FSM states
    // --------------------------------------------------------
    parameter IDLE        = 6'd0,
              STEP2_SQR   = 6'd1,
              STEP2_MUL   = 6'd2,
              STEP2_WAIT  = 6'd3,
              STEP3_SQR   = 6'd4,
              STEP3_MUL   = 6'd5,
              STEP3_WAIT  = 6'd6,
              STEP5_SQR   = 6'd7,
              STEP5_MUL   = 6'd8,
              STEP5_WAIT  = 6'd9,
              STEP10_SQR  = 6'd10,
              STEP10_MUL  = 6'd11,
              STEP10_WAIT = 6'd12,
              STEP20_SQR  = 6'd13,
              STEP20_MUL  = 6'd14,
              STEP20_WAIT = 6'd15,
              STEP40_SQR  = 6'd16,
              STEP40_MUL  = 6'd17,
              STEP40_WAIT = 6'd18,
              STEP80_SQR  = 6'd19,
              STEP80_MUL  = 6'd20,
              STEP80_WAIT = 6'd21,
              STEP81_SQR  = 6'd22,
              STEP81_MUL  = 6'd23,
              STEP81_WAIT = 6'd24,
              STEP162_SQR = 6'd25,
              STEP162_MUL = 6'd26,
              STEP162_WAIT= 6'd27,
              FINAL_SQR   = 6'd28,
              DONE        = 6'd29;
    
    reg [5:0] state;
    
    reg [7:0] sqr_count;
    reg [1:0] mul_count;

    // --------------------------------------------------------
    // FSM
    // --------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            inv       <= 0;
            done      <= 0;
            sqr_count <= 0;
            mul_count <= 0;
        end else begin
            case (state)

                // -------------------------------
                // IDLE
                // -------------------------------
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        b1    <= a;   // ?1 = a
                        state <= STEP2_SQR;
                    end
                end

                // -------------------------------
                // ?2 = (?1)^2 * ?1
                // -------------------------------
                STEP2_SQR: begin
                    sqr_in   <= b1;
                    state    <= STEP2_MUL;
                end
                STEP2_MUL: begin
                    mul_a    <= sqr_out;
                    mul_b    <= b1;
                    mul_count <= 0;
                    state    <= STEP2_WAIT;
                end
                STEP2_WAIT: begin
                    if (mul_count == 3) begin
                        b2    <= mul_res;
                        state <= STEP3_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?3 = (?2)^2 * ?1
                // -------------------------------
                STEP3_SQR: begin
                    sqr_in   <= b2;
                    state    <= STEP3_MUL;
                end
                STEP3_MUL: begin
                    mul_a    <= sqr_out;
                    mul_b    <= b1;
                    mul_count <= 0;
                    state    <= STEP3_WAIT;
                end
                STEP3_WAIT: begin
                    if (mul_count == 3) begin
                        b3    <= mul_res;
                        state <= STEP5_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?5 = (?3)^(2^2) * ?2
                // -------------------------------
                STEP5_SQR: begin
                    if (sqr_count < 2) begin
                        sqr_in   <= (sqr_count == 0) ? b3 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a     <= sqr_out;
                        mul_b     <= b2;
                        mul_count <= 0;
                        state     <= STEP5_WAIT;
                    end
                end
                STEP5_WAIT: begin
                    if (mul_count == 3) begin
                        b5    <= mul_res;
                        state <= STEP10_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?10 = (?5)^(2^5) * ?5
                // -------------------------------
                STEP10_SQR: begin
                    if (sqr_count < 5) begin
                        sqr_in   <= (sqr_count == 0) ? b5 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a     <= sqr_out;
                        mul_b     <= b5;
                        mul_count <= 0;
                        state     <= STEP10_WAIT;
                    end
                end
                STEP10_WAIT: begin
                    if (mul_count == 3) begin
                        b10   <= mul_res;
                        state <= STEP20_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?20 = (?10)^(2^10) * ?10
                // -------------------------------
                STEP20_SQR: begin
                    if (sqr_count < 10) begin
                        sqr_in   <= (sqr_count == 0) ? b10 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a     <= sqr_out;
                        mul_b     <= b10;
                        mul_count <= 0;
                        state     <= STEP20_WAIT;
                    end
                end
                STEP20_WAIT: begin
                    if (mul_count == 3) begin
                        b20   <= mul_res;
                        state <= STEP40_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?40 = (?20)^(2^20) * ?20
                // -------------------------------
                STEP40_SQR: begin
                    if (sqr_count < 20) begin
                        sqr_in   <= (sqr_count == 0) ? b20 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a     <= sqr_out;
                        mul_b     <= b20;
                        mul_count <= 0;
                        state     <= STEP40_WAIT;
                    end
                end
                STEP40_WAIT: begin
                    if (mul_count == 3) begin
                        b40   <= mul_res;
                        state <= STEP80_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?80 = (?40)^(2^40) * ?40
                // -------------------------------
                STEP80_SQR: begin
                    if (sqr_count < 40) begin
                        sqr_in   <= (sqr_count == 0) ? b40 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a     <= sqr_out;
                        mul_b     <= b40;
                        mul_count <= 0;
                        state     <= STEP80_WAIT;
                    end
                end
                STEP80_WAIT: begin
                    if (mul_count == 3) begin
                        b80   <= mul_res;
                        state <= STEP81_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?81 = (?80)^(2^1) * ?1
                // -------------------------------
                STEP81_SQR: begin
                    sqr_in   <= b80;
                    state    <= STEP81_MUL;
                end
                STEP81_MUL: begin
                    mul_a    <= sqr_out;
                    mul_b    <= b1;
                    mul_count <= 0;
                    state    <= STEP81_WAIT;
                end
                STEP81_WAIT: begin
                    if (mul_count == 3) begin
                        b81   <= mul_res;
                        state <= STEP162_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?162 = (?81)^(2^81) * ?81
                // -------------------------------
                STEP162_SQR: begin
                    if (sqr_count < 81) begin
                        sqr_in   <= (sqr_count == 0) ? b81 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a     <= sqr_out;
                        mul_b     <= b81;
                        mul_count <= 0;
                        state     <= STEP162_WAIT;
                    end
                end
                STEP162_WAIT: begin
                    if (mul_count == 3) begin
                        b162  <= mul_res;
                        state <= FINAL_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // Final inverse = (?162)^2
                // -------------------------------
                FINAL_SQR: begin
                    sqr_in <= b162;
                    state  <= DONE;
                end

                DONE: begin
                    inv  <= sqr_out;
                    done <= 1;
                    if (!start) begin
                        state <= IDLE;
                    end
                end
            endcase
        end
    end

endmodule
