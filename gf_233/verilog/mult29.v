`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 29-bit × 29-bit
// Polynomial multiplication, no reduction
// One-stage pipeline (registered output)
//////////////////////////////////////////////////////////////////////////////////
module mult29 (
    input  wire        clk,
    input  wire [28:0] a,
    input  wire [28:0] b,
    output reg  [57:0] d
);

    // partial products
    wire [57:0] pp [28:0];

    genvar i;
    generate
        for (i = 0; i < 29; i = i + 1) begin : gen_pp
            assign pp[i] = b[i] ? (a << i) : 58'b0;
        end
    endgenerate

    wire [57:0] d_comb;

    // XOR all partial products (combinational)
    assign d_comb = pp[0]  ^ pp[1]  ^ pp[2]  ^ pp[3]  ^ pp[4]  ^ pp[5]  ^
                    pp[6]  ^ pp[7]  ^ pp[8]  ^ pp[9]  ^ pp[10] ^ pp[11] ^
                    pp[12] ^ pp[13] ^ pp[14] ^ pp[15] ^ pp[16] ^ pp[17] ^
                    pp[18] ^ pp[19] ^ pp[20] ^ pp[21] ^ pp[22] ^ pp[23] ^
                    pp[24] ^ pp[25] ^ pp[26] ^ pp[27] ^ pp[28];

    // One pipeline stage: register the output
    always @(posedge clk) begin
        d <= d_comb;
    end

endmodule
