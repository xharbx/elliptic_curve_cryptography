`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 20-bit × 20-bit
// Polynomial multiplication, no reduction
// One-stage pipeline (registered output)
//////////////////////////////////////////////////////////////////////////////////
module mult20 (
    input  wire        clk,
    input  wire [19:0] a,
    input  wire [19:0] b,
    output reg  [39:0] d
);

    // Partial products
    wire [39:0] pp [19:0];

    genvar i;
    generate
        for (i = 0; i < 20; i = i + 1) begin : gen_pp
            assign pp[i] = b[i] ? (a << i) : 40'b0;
        end
    endgenerate

    wire [39:0] d_comb;

    // XOR all partial products
    assign d_comb = pp[0]  ^ pp[1]  ^ pp[2]  ^ pp[3]  ^ pp[4]  ^
                    pp[5]  ^ pp[6]  ^ pp[7]  ^ pp[8]  ^ pp[9]  ^
                    pp[10] ^ pp[11] ^ pp[12] ^ pp[13] ^ pp[14] ^
                    pp[15] ^ pp[16] ^ pp[17] ^ pp[18] ^ pp[19];

    // Pipeline stage
    always @(posedge clk) begin
        d <= d_comb;
    end

endmodule
