`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 21-bit × 21-bit
// Polynomial multiplication, no reduction
// One-stage pipeline (registered output)
//////////////////////////////////////////////////////////////////////////////////
module mult21 (
    input  wire        clk,
    input  wire [20:0] a,
    input  wire [20:0] b,
    output reg  [41:0] d
);

    // Partial products
    wire [41:0] pp [20:0];

    genvar i;
    generate
        for (i = 0; i < 21; i = i + 1) begin : gen_pp
            assign pp[i] = b[i] ? (a << i) : 42'b0;
        end
    endgenerate

    wire [41:0] d_comb;

    // XOR all partial products
    assign d_comb = pp[0]  ^ pp[1]  ^ pp[2]  ^ pp[3]  ^ pp[4]  ^ pp[5]  ^
                    pp[6]  ^ pp[7]  ^ pp[8]  ^ pp[9]  ^ pp[10] ^ pp[11] ^
                    pp[12] ^ pp[13] ^ pp[14] ^ pp[15] ^ pp[16] ^ pp[17] ^
                    pp[18] ^ pp[19] ^ pp[20];

    // One pipeline stage
    always @(posedge clk) begin
        d <= d_comb;
    end

endmodule
